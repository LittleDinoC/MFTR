import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPenaltyLayer(nn.Module):
    def __init__(self, init_tau=0.3, init_w_req=0.2, init_w_opt=0.1, alpha=15.0):
        super().__init__()
        self.raw_tau = nn.Parameter(torch.tensor(init_tau), requires_grad=True)
        self.raw_w_req = nn.Parameter(
            self._inverse_softplus(init_w_req), requires_grad=True
        )
        self.raw_w_opt = nn.Parameter(
            self._inverse_softplus(init_w_opt), requires_grad=True
        )
        self.alpha = alpha

    def _inverse_softplus(self, x):
        return torch.tensor(x).expm1().log()

    def forward(self, param_scores, is_required_mask):
        """
        Args:
            param_scores: [Batch_size, Max_Params], 所有参数的最佳匹配分
            is_required_mask: [Batch_size, Max_Params], 1.0 for required, 0.0 for optional/padding

        Returns:
            total_penalty: [Batch_size], 每个样本的总惩罚分数
        """
        tau = torch.clamp(self.raw_tau, 0.0, 1.0)
        w_req = F.softplus(self.raw_w_req)
        w_opt = F.softplus(self.raw_w_opt)

        soft_count = torch.sigmoid(self.alpha * (tau - param_scores))

        is_optional_mask = 1.0 - is_required_mask

        penalty_weights = (is_required_mask * w_req) + (is_optional_mask * w_opt)

        penalty_score = soft_count * penalty_weights
        total_penalty = torch.sum(penalty_score, dim=-1)

        return total_penalty, tau, w_req, w_opt


class ToolRanker(nn.Module):
    def __init__(self, num_base_features, soft_penalty_config={}):
        super().__init__()

        self.base_score_layer = nn.Linear(num_base_features, 1)

        self.soft_penalty = SoftPenaltyLayer(**soft_penalty_config)

    def forward(self, base_features, param_scores, is_required_mask):
        """
        Args:
            base_features: [Batch_size, num_base_features], 包含 Description Score 等
            param_scores: [Batch_size, Max_Params], 参数匹配分
            is_required_mask: [Batch_size, Max_Params], 是否是必选参数
        """
        base_score = self.base_score_layer(base_features)
        total_penalty, tau, w_req, w_opt = self.soft_penalty(
            param_scores, is_required_mask
        )
        score_final = base_score.squeeze(-1) - total_penalty
        return score_final, tau, w_req, w_opt
