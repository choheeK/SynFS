
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils.mlflow_logger import MLflowLogger

class SynFSTrainer:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.logger = MLflowLogger(cfg)

        self.model = model
        self.device = cfg.device

        self.loss = nn.CrossEntropyLoss(reduction="none")

        # optimizers exactly as original code
        self.opt_h = torch.optim.Adam(
            list(self.model.s_model.shared_predictor.parameters()) +
            list(self.model.ns_model.shared_predictor.parameters()),
            lr=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
        )

        self.opt_s = torch.optim.Adam(
            [p for sel in self.model.s_model.s_selectors for p in sel.parameters()],
            lr=cfg.model.s_learning_rate,
        )

        self.opt_ns = torch.optim.Adam(
            [p for sel in self.model.ns_model.s_selectors for p in sel.parameters()],
            lr=cfg.model.s_learning_rate,
        )

        self.opt_allinf = torch.optim.Adam(
            [p for sel in self.model.s_model.s_selectors for p in sel.parameters()] +
            [p for sel in self.model.ns_model.s_selectors for p in sel.parameters()] +
            list(self.model.all_inf.parameters()),
            lr=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
        )

        self.X_mean_set = None

    # ---------------------------------------------------------
    # Compute GLOBAL mean of each view (original code behavior)
    # ---------------------------------------------------------
    def compute_view_means_from_loader(self, loader):
        """
        Computes global mean of each view by iterating once over the loader.
        Equivalent to original code using full tr_X_set.
        """
        n_views = len(self.cfg.model.views_dims)
        sums = [0] * n_views
        counts = 0

        for views, _ in loader:
            views = [v.to(self.device) for v in views]
            batch_size = views[0].shape[0]

            for i in range(n_views):
                sums[i] = sums[i] + views[i].sum(0)
            counts += batch_size

        return [s / counts for s in sums]

    def set_X_mean_set(self, train_loader):
        self.X_mean_set = self.compute_view_means_from_loader(train_loader)

    # ---------------------------------------------------------
    # Mask generator (unchanged from original logic)
    # ---------------------------------------------------------
    def mask_generator(self, batch_size):
        masks = []
        dims = self.cfg.model.views_dims
        total = sum(dims)

        cumsum = torch.tensor(dims).cumsum(0)
        blank = torch.zeros(batch_size, total, device=self.device)

        for i in range(len(dims)):
            m = blank.clone()
            if i == 0:
                m[:, :cumsum[i]] = 1
            else:
                m[:, cumsum[i - 1]:cumsum[i]] = 1
            masks.append(m)

        return masks

    def reg(self, selector):
        return torch.mean(selector.regularizer(selector.mu / selector.sigma))


    def train_step(self, batch):

        if self.X_mean_set is None:
            raise RuntimeError("X_mean_set missing. Call set_X_mean_set() first.")

        X_mean_set = self.X_mean_set

        # -----------------------------
        # Prepare data
        # -----------------------------
        views, y = batch
        y = y.to(self.device)
        views = [v.to(self.device) for v in views]

        batch_size = y.size(0)
        masks = self.mask_generator(batch_size)

        # =========================================================================
        # 1) SELECTOR FORWARD PASS (same noise behavior as original)
        # =========================================================================
        v_s, z_s = [], []
        for i, sel in enumerate(self.model.s_model.s_selectors):
            out = sel(views[i], X_mean_set[i])
            v_s.append(out)
            z_s.append(sel.z)

        v_n, z_n = [], []
        for i, sel in enumerate(self.model.ns_model.s_selectors):
            out = sel(views[i], X_mean_set[i])
            v_n.append(out)
            z_n.append(sel.z)
        # helper forward
        def fwd(model, vlist):
            full = torch.cat(vlist, dim=1)
            outs = [model.shared_predictor(full * m) for m in masks]
            outs.append(model.shared_predictor(full))
            return outs

        # =========================================================================
        # 2) PREDICTOR UPDATE (p_loss)  — EXACT SAME AS ORIGINAL
        # =========================================================================
        n_logits = fwd(self.model.ns_model, v_n)
        s_logits = fwd(self.model.s_model, v_s)

        n_losses = [self.loss(l, y) for l in n_logits]
        s_losses = [self.loss(l, y) for l in s_logits]

        p_loss = torch.mean(torch.sum(torch.stack(n_losses + s_losses, dim=1), dim=1))

        self.opt_h.zero_grad()
        p_loss.backward(retain_graph=True)
        self.opt_h.step()

        # =========================================================================
        # 3) Synergistic Selector & Predictor 
        # =========================================================================
        # ALL-INF gate = max(hard_sigmoid(z_s), hard_sigmoid(z_n))
        all_gates = [
            self.model.s_model.s_selectors[0].hard_sigmoid(torch.max(s, n))
            for s, n in zip(z_s, z_n)
        ]

        all_v = [
            views[i] * all_gates[i] + X_mean_set[i] * (1 - all_gates[i])
            for i in range(len(views))
        ]

        # use original hard_sigmoid
        gate_s = [self.model.s_model.s_selectors[0].hard_sigmoid(z) for z in z_s]
        gate_n = [self.model.s_model.s_selectors[0].hard_sigmoid(z) for z in z_n]

        s_regs = [self.reg(sel) for sel in self.model.s_model.s_selectors]

        # inf-loss
        all_inf_logits = self.model.all_inf(torch.cat(all_v, dim=1))
        inf_loss = torch.mean(self.loss(all_inf_logits, y))

        # Synergy loss: v_bar - masked_sum + reg
        s_logits = fwd(self.model.s_model, v_s)
        s_v_bar = s_logits[-1]
        s_masked = s_logits[:-1]


        synergy_loss = torch.mean(
            self.loss(s_v_bar, y)
            - torch.sum(
                torch.stack([self.loss(l, y) for l in s_masked], dim=1),
                dim=1,
            )
            + self.cfg.model.s_lam * torch.mean(torch.stack(s_regs))
        )

        self.opt_allinf.zero_grad()
        self.opt_s.zero_grad()
        inf_loss.backward(retain_graph=True)
        synergy_loss.backward()
        self.opt_allinf.step()
        self.opt_s.step()

        # =========================================================================
        # 4) NON-SYNERGISTIC SELECTOR UPDATE
        # =========================================================================


        sim = torch.nn.functional.cosine_similarity(
            torch.cat(gate_s, dim=1),
            torch.cat(gate_n, dim=1),
            dim=1
        )

        ns_regs = [self.reg(sel) for sel in self.model.ns_model.s_selectors]

        n_logits = fwd(self.model.ns_model, v_n)
        n_v_bar = n_logits[-1]
        n_masked = n_logits[:-1]

        nsynergy_loss = torch.mean(
            -self.loss(n_v_bar, y)
            + torch.sum(
                torch.stack([self.loss(l, y) for l in n_masked], dim=1),
                dim=1,
            )
            + self.cfg.model.ns_lam * torch.mean(torch.stack(ns_regs))
            + self.cfg.model.ns_alpha * sim
        )

        self.opt_ns.zero_grad()
        nsynergy_loss.backward()
        self.opt_ns.step()

        # =========================================================================
        # 5) AUROC (same return order as original)
        # =========================================================================
        try:
            auroc = roc_auc_score(
                y.detach().cpu().numpy(),
                all_inf_logits.detach().cpu().numpy()[:, 1]
            )
        except:
            auroc = 0.0

        return auroc, inf_loss.item(), synergy_loss.item(), nsynergy_loss.item()

    @torch.no_grad()
    def predict(self, views):
        #t_index to check the target view , if none logits will be the same 
        self.model.eval()
        
        if self.X_mean_set is None:
            raise RuntimeError("X_mean_set is missing. Call trainer.set_X_mean_set(train_loader) first.")

        X_mean_set = self.X_mean_set

        with torch.no_grad():
            S = self.model.get_detached_mu(self.model.s_model)
            NS = self.model.get_detached_mu(self.model.ns_model)
            all_mu = [torch.max(s, ns) for s, ns in zip(S, NS)]
            all_z = [all_mu[i]*views[i]+(1-all_mu[i])*X_mean_set[i] for i in range(len(views))]
            all_bar_logits = self.model.all_inf(torch.cat(all_z, dim=1))

        return all_bar_logits


    @torch.no_grad()
    def evaluate(self, batch):
        """
        Exact reproduction of original SynFS.evaluate().
        Returns:
            ai_auroc, n_auroc, s_auroc, s_gate, ns_gate,
            all_inf_loss, nsynergy_loss, synergy_loss
        """
        if self.X_mean_set is None:
            raise RuntimeError("X_mean_set is missing. Call trainer.set_X_mean_set(train_loader) first.")

        X_mean_set = self.X_mean_set

        views, y = batch
        y = y.to(self.device)
        views = [v.to(self.device) for v in views]

        batch_size = y.shape[0]
        masks = self.mask_generator(batch_size)

        # ----------------------------------------------------
        # 1. Compute μ-based (detached) gates without noise
        # ----------------------------------------------------
        S = self.model.get_detached_mu(self.model.s_model)
        NS = self.model.get_detached_mu(self.model.ns_model)

        # all-gate = max(s, ns)
        all_mu = [torch.max(s, ns) for s, ns in zip(S, NS)]

        # Replace missing features using global mean
        all_v = [
            all_mu[i] * views[i] + (1 - all_mu[i]) * X_mean_set[i]
            for i in range(len(views))
        ]

        # ----------------------------------------------------
        # 2. Compute all-inf logits
        # ----------------------------------------------------
        all_inf_logits = self.model.all_inf(torch.cat(all_v, dim=1))

        # ----------------------------------------------------
        # 3. Forward for masked/unmasked predictors
        # ----------------------------------------------------
        def eval_forward(model, gates):
            v_list = [
                gates[i] * views[i] + (1 - gates[i]) * X_mean_set[i]
                for i in range(len(views))
            ]
            concat = torch.cat(v_list, dim=1)

            logits = []
            for m in masks:
                logits.append(model.shared_predictor(concat * m))

            logits.append(model.shared_predictor(concat))
            return logits

        s_logits = eval_forward(self.model.s_model, S)
        n_logits = eval_forward(self.model.ns_model, NS)

        # losses
        s_losses = [self.loss(l, y) for l in s_logits]
        n_losses = [self.loss(l, y) for l in n_logits]

        all_inf_loss = torch.mean(self.loss(all_inf_logits, y))

        # synergy
        s_v_bar_loss = s_losses[-1]
        synergy_loss = torch.mean(
            s_v_bar_loss - torch.sum(torch.stack(s_losses[:-1], dim=1), dim=1)
        )

        # non-synergy
        n_v_bar_loss = n_losses[-1]
        nsynergy_loss = -torch.mean(
            n_v_bar_loss - torch.sum(torch.stack(n_losses[:-1], dim=1), dim=1)
        )

        # ----------------------------------------------------
        # 4. Compute AUROC
        # ----------------------------------------------------
        try:
            ai_auroc = roc_auc_score(
                y.cpu().numpy(), all_inf_logits.cpu().numpy()[:, 1]
            )
            n_auroc = roc_auc_score(
                y.cpu().numpy(), n_logits[-1].cpu().numpy()[:, 1]
            )
            s_auroc = roc_auc_score(
                y.cpu().numpy(), s_logits[-1].cpu().numpy()[:, 1]
            )
        except:
            ai_auroc, n_auroc, s_auroc = 0.0, 0.0, 0.0

        return (
            float(ai_auroc),
            float(n_auroc),
            float(s_auroc),
            torch.cat(S),
            torch.cat(NS),
            float(all_inf_loss),
            float(nsynergy_loss),
            float(synergy_loss),
        )
    def train_epoch(self, train_loader):
        self.model.train()

        
        losses = []
        aurocs = []

        for batch in train_loader:
            auroc, a_loss, s_loss, n_loss = self.train_step(batch)
            aurocs.append(float(auroc))
            losses.append(float(a_loss + s_loss + n_loss))

        return {
            "loss": float(np.mean(losses)),
            "auroc": float(np.mean(aurocs)),
        }


    # =====================================================================
    # VALIDATE ONE EPOCH
    # =====================================================================
    @torch.no_grad()
    def validate_epoch(self, val_loader):
        self.model.eval()
        losses = []
        aurocs = []

        for batch in val_loader:
            ai_auroc, n_auroc, s_auroc, s_gate, ns_gate, a_loss, n_loss, s_loss = \
                self.evaluate(batch)

            aurocs.append(float(ai_auroc))
            losses.append(float(a_loss + s_loss + n_loss))

        return {
            "loss": float(np.mean(losses)),
            "auroc": float(np.mean(aurocs)),
        }


    # =====================================================================
    # TOP-LEVEL TRAIN LOOP
    # =====================================================================
    def train(self, train_loader, val_loader=None):

        # compute dataset-level mean ONCE (correct SynFS behavior)
        self.set_X_mean_set(train_loader)
        for epoch in range(self.cfg.nr_epochs):
            train_metrics = self.train_epoch(train_loader)
            print(f"[Epoch {epoch+1}] Train AUROC = {train_metrics['auroc']:.4f}")
            
            self.logger.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_auroc": train_metrics["auroc"],
                },
                step=epoch
            )



            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                print(f"[Epoch {epoch+1}] Val AUROC   = {val_metrics['auroc']:.4f}")

                self.logger.log_metrics(
                {
                    "val_loss": val_metrics["loss"],
                    "val_auroc": val_metrics["auroc"],
                },
                step=epoch
                )

