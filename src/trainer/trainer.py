import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from models.synfs_model import SynFSModel
from utils.seed import fix_seed
from utils.mlflow_logger import MLflowLogger

class SynFSTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.model = SynFSModel(cfg.model).to(self.device)
        self.loss = nn.CrossEntropyLoss(reduction="none")

        # Collect params
        s_h_params = list(self.model.s_model.shared_predictor.parameters())
        ns_h_params = list(self.model.ns_model.shared_predictor.parameters())

        s_params = [p for sel in self.model.s_model.s_selectors for p in sel.parameters()]
        ns_params = [p for sel in self.model.ns_model.s_selectors for p in sel.parameters()]

        self.opt_h = torch.optim.Adam(
            s_h_params + ns_h_params,
            lr=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay
        )

        self.opt_s = torch.optim.Adam(s_params, lr=cfg.model.s_learning_rate)
        self.opt_ns = torch.optim.Adam(ns_params, lr=cfg.model.s_learning_rate)

        all_params = s_params + ns_params + list(self.model.all_inf.parameters())
        self.opt_allinf = torch.optim.Adam(
            all_params,
            lr=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay
        )

    # ---------- helper ----------
    def mask_generator(self, batch):
        dims = self.cfg.model.views_dims
        total = sum(dims)
        masks = []
        cumsum = np.cumsum(dims)

        base = torch.zeros(batch, total).to(self.device)
        for i in range(len(dims)):
            m = base.clone()
            start = 0 if i == 0 else cumsum[i-1]
            end = cumsum[i]
            m[:, start:end] = 1
            masks.append(m)
        return masks

    def _compute_regularizer(self, selector):
        reg_fn = selector.regularizer
        return torch.mean(reg_fn(selector.mu / selector.sigma))


    # ---------- train ----------
    def train_step(self, batch):
        """
        One training iteration. Fully rewritten version of your original logic
        with identical mathematical behavior.

        Returns:
            dict: { "loss": float, "auroc": float }
        """
        views, y = batch
        y = y.to(self.device)

        # Move views to device
        views = [v.to(self.device) for v in views]

        batch_size = y.size(0)
        masks = self.mask_generator(batch_size)

        # ============================================================
        # 1) Forward pass through selectors (with noise)
        # ============================================================
        v_s = [sel(views[i], views[i].mean(0)) for i, sel in enumerate(self.model.s_model.s_selectors)]
        v_n = [sel(views[i], views[i].mean(0)) for i, sel in enumerate(self.model.ns_model.s_selectors)]

        # z-values (before sigmoid), also contain noise
        z_s = [sel.z for sel in self.model.s_model.s_selectors]
        z_n = [sel.z for sel in self.model.ns_model.s_selectors]

        # ============================================================
        # 2) Predictor Phase: h_loss (shared predictor update)
        # ============================================================
        def forward_views(model, v_list):
            logits = []
            concat = torch.cat(v_list, dim=1)
            for m in masks:
                logits.append(model.shared_predictor(concat * m))
            logits.append(model.shared_predictor(concat))  # full-view
            return logits

        n_logits = forward_views(self.model.ns_model, v_n)
        s_logits = forward_views(self.model.s_model, v_s)

        n_losses = [self.loss(logit, y) for logit in n_logits]
        s_losses = [self.loss(logit, y) for logit in s_logits]

        # h_loss = mean over batch( sum over views(n,s losses) )
        h_loss = torch.mean(torch.sum(torch.stack(n_losses + s_losses, dim=1), dim=1))

        # Optimize (shared predictor only)
        self.opt_h.zero_grad()
        h_loss.backward(retain_graph=True)
        self.opt_h.step()

        # ============================================================
        # 3) Synergistic selector + All-Informative predictor
        # ============================================================

        # gates WITH noise (max(z_s, z_n))
        all_gates = [
            self.model.s_model.s_selectors[0].hard_sigmoid(torch.max(s, n))
            for s, n in zip(z_s, z_n)
        ]

        # fill missing values with view-wise mean
        X_means = [v.mean(0) for v in views]
        all_v = [views[i] * all_gates[i] + X_means[i] * (1 - all_gates[i])
                 for i in range(len(views))]

        # regularizers
        s_regs = [self._compute_regularizer(sel) for sel in self.model.s_model.s_selectors]

        # all-inf logit
        all_inf_logits = self.model.all_inf(torch.cat(all_v, dim=1))
        all_inf_loss = torch.mean(self.loss(all_inf_logits, y))

        # Synergy: full predictor minus sum of masked predictors
        s_logits = forward_views(self.model.s_model, v_s)
        s_full = s_logits[-1]
        s_losses = [self.loss(logit, y) for logit in s_logits[:-1]]
        s_full_loss = self.loss(s_full, y)

        synergy_loss = torch.mean(
            s_full_loss
            - torch.sum(torch.stack(s_losses, dim=1), dim=1)
            + self.cfg.model.s_lam * torch.mean(torch.stack(s_regs))
        )

        # optimize synergy & all-informative
        self.opt_allinf.zero_grad()
        self.opt_s.zero_grad()

        all_inf_loss.backward(retain_graph=True)
        synergy_loss.backward()

        self.opt_allinf.step()
        self.opt_s.step()

        # ============================================================
        # 4) Non-synergistic selector loss (repulsion term)
        # ============================================================

        gate_s = [torch.sigmoid(z) for z in z_s]
        gate_n = [torch.sigmoid(z) for z in z_n]

        # cosine similarity penalty
        sim = torch.nn.functional.cosine_similarity(
            torch.cat(gate_s, dim=1),
            torch.cat(gate_n, dim=1),
            dim=1
        )

        # regularizers for non-synergy
        ns_regs = [self._compute_regularizer(sel) for sel in self.model.ns_model.s_selectors]

        # compute non-synergy predictor outputs
        n_logits = forward_views(self.model.ns_model, v_n)
        n_full = n_logits[-1]
        n_losses = [self.loss(logit, y) for logit in n_logits[:-1]]
        n_full_loss = self.loss(n_full, y)

        nsynergy_loss = torch.mean(
            -n_full_loss
            + torch.sum(torch.stack(n_losses, dim=1), dim=1)
            + self.cfg.model.ns_lam * torch.mean(torch.stack(ns_regs))
            + self.cfg.model.ns_alpha * sim
        )

        # optimize non-synergistic
        self.opt_ns.zero_grad()
        nsynergy_loss.backward()
        self.opt_ns.step()

        # ============================================================
        # 5) Compute AUROC
        # ============================================================

        try:
            ai = roc_auc_score(y.cpu().numpy(), all_inf_logits.detach().cpu().numpy()[:, 1])
            sn = roc_auc_score(y.cpu().numpy(), s_full.detach().cpu().numpy()[:, 1])
            nn = roc_auc_score(y.cpu().numpy(), n_full.detach().cpu().numpy()[:, 1])
            auroc = float(ai)
        except:
            auroc = 0.0

        # ============================================================
        # 6) Total loss (for logging only)
        # ============================================================
        total_loss = (
            float(h_loss.detach().cpu())
            + float(all_inf_loss.detach().cpu())
            + float(synergy_loss.detach().cpu())
            + float(nsynergy_loss.detach().cpu())
        ) / 4.0

        return {
            "loss": total_loss,
            "auroc": auroc,
        }

    def train(self, train_loader, val_loader):
        logger = MLflowLogger(self.cfg)

        for epoch in range(self.cfg.nr_epochs):
            train_metrics = self.run_epoch(train_loader)
            val_metrics   = self.run_val(val_loader)

            logger.log_metrics(train_metrics, epoch, prefix="train")
            logger.log_metrics(val_metrics, epoch, prefix="val")

        logger.save_model(self.model, "model")
        logger.end()

    def run_epoch(self, loader):
        self.model.train()
        metrics_list = []

        for batch in loader:
            metrics = self.train_step(batch)
            metrics_list.append(metrics["loss"])

        return {"loss": float(np.mean(metrics_list))}

    def run_val(self, loader):
        """
        Validation loop

        Returns dict:
            {
                "loss": float,
                "auroc": float,
                "s_auroc": float,
                "n_auroc": float
            }
        """

        self.model.eval()

        all_ai = []
        all_s = []
        all_n = []
        all_losses = []

        for batch in loader:
            views, y = batch
            y = y.to(self.device)
            views = [v.to(self.device) for v in views]
            batch_size = y.size(0)

            # ------------------------------------------------------------
            # 1. Compute gates WITHOUT gradient (detached mu)
            # ------------------------------------------------------------
            S  = [sel.hard_sigmoid(sel.mu.detach()) for sel in self.model.s_model.s_selectors]
            NS = [sel.hard_sigmoid(sel.mu.detach()) for sel in self.model.ns_model.s_selectors]

            # all-inf gate = max(S, NS)
            ALL = [torch.max(s, n) for s, n in zip(S, NS)]

            # means for missing fill
            X_means = [v.mean(0) for v in views]

            # ------------------------------------------------------------
            # 2. Build all-inf views
            # ------------------------------------------------------------
            all_z = [
                ALL[i] * views[i] + (1 - ALL[i]) * X_means[i]
                for i in range(len(views))
            ]

            # all-inf logits
            all_bar_logits = self.model.all_inf(torch.cat(all_z, dim=1))
            ai_loss = torch.mean(self.loss(all_bar_logits, y))

            # ------------------------------------------------------------
            # 3. Build synergy and non-synergy logits for all views
            # ------------------------------------------------------------
            def forward_views(model, gates):
                """Helper to compute masked+full logits."""
                masks = self.mask_generator(batch_size)
                z_list = [g * v + (1 - g) * xm for g, v, xm in zip(gates, views, X_means)]
                concat = torch.cat(z_list, dim=1)

                logits = []
                for m in masks:
                    logits.append(model.shared_predictor(concat * m))

                full = model.shared_predictor(concat)
                logits.append(full)
                return logits

            s_logits = forward_views(self.model.s_model, S)
            n_logits = forward_views(self.model.ns_model, NS)

            # ------------------------------------------------------------
            # 4. Compute losses EXACTLY like original code
            # ------------------------------------------------------------
            # synergy losses
            s_losses = [self.loss(l, y) for l in s_logits]
            s_full = s_losses[-1]
            s_partial = s_losses[:-1]

            synergy_loss = torch.mean(
                s_full - torch.sum(torch.stack(s_partial, dim=1), dim=1)
            )

            # non-synergy losses
            n_losses = [self.loss(l, y) for l in n_logits]
            n_full = n_losses[-1]
            n_partial = n_losses[:-1]

            nsynergy_loss = torch.mean(
                -(n_full - torch.sum(torch.stack(n_partial, dim=1), dim=1))
            )

            # ------------------------------------------------------------
            # 5. Compute AUROCs
            # ------------------------------------------------------------
            try:
                ai_auroc = roc_auc_score(y.cpu().numpy(), all_bar_logits.cpu().numpy()[:, 1])
                s_auroc  = roc_auc_score(y.cpu().numpy(), s_logits[-1].cpu().numpy()[:, 1])
                n_auroc  = roc_auc_score(y.cpu().numpy(), n_logits[-1].cpu().numpy()[:, 1])
            except:
                ai_auroc = 0.0
                s_auroc  = 0.0
                n_auroc  = 0.0

            # collect
            all_ai.append(ai_auroc)
            all_s.append(s_auroc)
            all_n.append(n_auroc)
            all_losses.append(float(ai_loss))

        # ============================================================
        # 6. Return aggregated metrics
        # ============================================================
        return {
            "loss": float(np.mean(all_losses)),
            "auroc": float(np.mean(all_ai)),
            "s_auroc": float(np.mean(all_s)),
            "n_auroc": float(np.mean(all_n)),
        }
        return {"loss": 0.0}
