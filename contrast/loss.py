import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, gamma=2, pos_ratio=0.7, margin=100.):
        super(TripletLoss, self).__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.gamma = gamma
        self.pos_ratio = pos_ratio

    def regression_loss(self, q, k, coord_q, coord_k):
        """ q, k: N * C * H * W
            coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
        """
        Nq, Cq, Hq, Wq = q.shape
        Nk, Ck, Hk, Wk = k.shape

        # [bs, feat_dim, 49]
        q = q.view(Nq, Cq, -1)
        k = k.view(Nk, Ck, -1)

        # generate center_coord, width, height
        # [1, 7, 7]
        qx_array = torch.arange(0., float(
            Wq), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1).repeat(1, Hq, 1)
        qy_array = torch.arange(0., float(
            Hq), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1).repeat(1, 1, Wq)

        kx_array = torch.arange(0., float(
            Wk), dtype=coord_k.dtype, device=coord_k.device).view(1, 1, -1).repeat(1, Hk, 1)
        ky_array = torch.arange(0., float(
            Hk), dtype=coord_k.dtype, device=coord_k.device).view(1, -1, 1).repeat(1, 1, Wk)

        # [bs, 1, 1]
        q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / Wq).view(-1, 1, 1)
        q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / Hq).view(-1, 1, 1)
        k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / Wk).view(-1, 1, 1)
        k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / Hk).view(-1, 1, 1)
        # [bs, 1, 1]
        q_start_x = coord_q[:, 0].view(-1, 1, 1)
        q_start_y = coord_q[:, 1].view(-1, 1, 1)
        k_start_x = coord_k[:, 0].view(-1, 1, 1)
        k_start_y = coord_k[:, 1].view(-1, 1, 1)

        # [bs, 1, 1]
        q_bin_diag = torch.sqrt(q_bin_width ** 2 + q_bin_height ** 2)
        k_bin_diag = torch.sqrt(k_bin_width ** 2 + k_bin_height ** 2)
        max_bin_diag = torch.max(q_bin_diag, k_bin_diag)

        # [bs, 7, 7]
        center_q_x = (qx_array + 0.5) * q_bin_width + q_start_x
        center_q_y = (qy_array + 0.5) * q_bin_height + q_start_y
        center_k_x = (kx_array + 0.5) * k_bin_width + k_start_x
        center_k_y = (ky_array + 0.5) * k_bin_height + k_start_y

        # [bs, 49, 49]
        dist_center = torch.sqrt((center_q_x.view(-1, Hq * Wq, 1) - center_k_x.view(-1, 1, Hk * Wk)) ** 2
                                 + (center_q_y.view(-1, Hq * Wq, 1) - center_k_y.view(-1, 1, Hk * Wk)) ** 2) / max_bin_diag
        pos_mask = (dist_center < self.pos_ratio).float().detach()

        # [bs, 49, 49]
        logits = -2. * torch.bmm(q.transpose(1, 2), k)

        positives = (logits * pos_mask).sum(-1).sum(-1) / \
            (pos_mask.sum(-1).sum(-1) + 1e-6)

        masked_logits = logits.masked_fill_(pos_mask.bool(), float("inf"))
        negatives, _ = torch.topk(masked_logits, 10, dim=-1, largest=False)
        negatives = negatives[:, :, 1:].mean(-1).mean(-1)

        return self.ranking_loss(negatives, self.gamma * positives, torch.ones_like(positives))

    def forward(self, gl_sa_maps, lo_sa_maps, gl_projs_ng, coords):

        gl_coords = coords[:2]
        lo_coords = coords[2:]

        gl_sa_maps = gl_sa_maps.chunk(2)
        gl_projs_ng = gl_projs_ng.chunk(2)
        lo_sa_maps = lo_sa_maps.chunk(6)

        ctx2loc_loss = 0
        n_terms = 0

        for j in range(len(gl_projs_ng)):
            for i in range(len(gl_sa_maps)):
                if i == j:
                    continue
                # ctx to local loss on global views
                ctx2loc_loss += self.regression_loss(
                    gl_sa_maps[i], gl_projs_ng[j], gl_coords[i], gl_coords[j])
                n_terms += 1

            for i in range(len(lo_sa_maps)):
                # ctx to local loss on local views
                ctx2loc_loss += self.regression_loss(
                    lo_sa_maps[i], gl_projs_ng[j], lo_coords[i], gl_coords[j])
                n_terms += 1

        ctx2loc_loss /= n_terms
        return ctx2loc_loss
