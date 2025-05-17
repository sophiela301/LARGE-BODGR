import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """User preference encoder"""

    def __init__(self, n_items, emb_dim, drop_ratio):
        super(Encoder, self).__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.drop = nn.Dropout(drop_ratio)

        self.user_preference_encoder = nn.Linear(
            self.n_items, self.emb_dim, bias=True)
        nn.init.xavier_uniform_(self.user_preference_encoder.weight)
        nn.init.zeros_(self.user_preference_encoder.bias)

        self.transform_layer = nn.Linear(self.emb_dim, self.emb_dim)
        nn.init.xavier_uniform_(self.transform_layer.weight)
        nn.init.zeros_(self.transform_layer.bias)

        self.user_predictor = nn.Linear(self.emb_dim, self.n_items, bias=False)
        nn.init.xavier_uniform_(self.user_predictor.weight)

    def pretrain_forward(self, user_items):
        user_items_norm = F.normalize(user_items)
        user_pref_emb = self.drop(user_items_norm)
        user_pref_emb = torch.tanh(self.user_preference_encoder(user_pref_emb))

        logits = self.user_predictor(user_pref_emb)
        return logits, user_pref_emb

    def forward(self, user_items):
        _, user_embeds = self.pretrain_forward(user_items)
        user_embeds = torch.tanh(self.transform_layer(user_embeds))
        return user_embeds


class AttentionAggregator(nn.Module):
    """User preferences aggregator (attention-based)"""

    def __init__(self, output_dim, drop_ratio=0.):
        super(AttentionAggregator, self).__init__()

        self.attention = nn.Linear(output_dim, 1)
        self.drop = nn.Dropout(drop_ratio)
        self.classifier = ComplexMLPClassifier(4, [128, 64], 2, 0.5)

    def pre_forward(self, x, mask):
        attention_out = torch.tanh(self.attention(x))
        if mask is None:
            weight = torch.softmax(attention_out, dim=1)
        else:
            weight = torch.softmax(attention_out + mask.unsqueeze(2), dim=1)

        ret = torch.matmul(x.transpose(2, 1), weight).squeeze(2)
        return ret, weight, mask

    def forward(self, x, mask):
        attention_out = torch.tanh(self.attention(x))
        if mask is None:
            weight = torch.softmax(attention_out, dim=1)
        else:
            weight = torch.softmax(attention_out + mask.unsqueeze(2), dim=1)
        # soft----------------------------------------------------
        h = x
        max_weightmemver_indices = torch.argmax(weight, dim=1)
        predicted_classes = self.classifier(weight)
        predicted_classes[:, 1] = predicted_classes[:, 1]-0.5  # bias
        predicted_classes = torch.argmax(predicted_classes, dim=1)
        ret = torch.zeros(x.shape[0], x.shape[2])
        high_variance_indices = []
        for i in range(len(predicted_classes)):
            if predicted_classes[i] == 1:  # leadership
                ret[i] = h[i][max_weightmemver_indices[i]]
                high_variance_indices.append(i)
            else:
                ret[i] = torch.sum(weight[i] * h[i], dim=0)
        # ret = torch.matmul(x.transpose(2, 1), weight).squeeze(2)
        return ret, weight, predicted_classes


class Discriminator(nn.Module):
    """Discriminator - bilinear layer outputs the user-group score"""

    def __init__(self, emb_dim=64):
        super(Discriminator, self).__init__()
        self.emb_dim = emb_dim

        self.fc_layer = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        nn.init.xavier_uniform_(self.fc_layer.weight)
        nn.init.zeros_(self.fc_layer.bias)

        self.bilinear_layer = nn.Bilinear(self.emb_dim, self.emb_dim, 1)
        nn.init.xavier_uniform_(self.bilinear_layer.weight)
        nn.init.zeros_(self.bilinear_layer.bias)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, group_inputs, user_inputs):
        group_emb = torch.tanh(self.fc_layer(group_inputs))

        user_emb = torch.tanh(self.fc_layer(user_inputs))

        return self.bilinear_layer(user_emb, group_emb.unsqueeze(1).repeat(1, user_inputs.shape[1], 1))

    def mi_loss(self, scores_group, group_mask, scores_corrupted, device="cpu"):
        batch_size = scores_group.shape[0]

        pos_size, neg_size = scores_group.shape[1], scores_corrupted.shape[1]

        one_labels = torch.ones(batch_size, pos_size).to(device)
        zero_labels = torch.zeros(batch_size, neg_size).to(device)

        labels = torch.cat((one_labels, zero_labels), 1)
        logits = torch.cat((scores_group, scores_corrupted), 1).squeeze(2)

        mask = torch.cat((torch.exp(group_mask), torch.ones(
            [batch_size, neg_size]).to(device)), 1)

        mi_loss = self.bce_loss(logits * mask, labels * mask) * (batch_size * (pos_size + neg_size)) \
            / (torch.exp(group_mask).sum() + batch_size * neg_size)

        return mi_loss


class GroupIM(nn.Module):
    def __init__(self, n_items, emb_dim, lambda_mi=0.1, drop_ratio=0.4):
        super(GroupIM, self).__init__()
        self.n_items = n_items
        self.lambda_mi = lambda_mi
        self.drop = nn.Dropout(drop_ratio)

        self.emb_dim = emb_dim
        self.classifier_threshold = 0.5  
        self.user_preference_encoder = Encoder(
            self.n_items, self.emb_dim, drop_ratio)
        self.preference_aggregator = AttentionAggregator(self.emb_dim)
        self.group_predictor = nn.Linear(
            self.emb_dim, self.n_items, bias=False)
        nn.init.xavier_uniform_(self.group_predictor.weight)

        self.discriminator = Discriminator(emb_dim=self.emb_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, group_mask, user_items):
        user_pref_embeds = self.user_preference_encoder(user_items)
        group_embed, weights, targets = self.preference_aggregator(
            user_pref_embeds, group_mask)
        group_logit = self.group_predictor(group_embed.cuda(0))

        if self.train:
            obs_user_embeds = self.user_preference_encoder(user_items)
            scores_ug = self.discriminator(
                group_embed.cuda(0), obs_user_embeds).detach()
            return group_logit, group_embed.cuda(0), scores_ug, weights, targets
        else:
            return group_logit, group_embed.cuda(0)

    def pre_forward(self, group_mask, user_items):
        user_pref_embeds = self.user_preference_encoder(user_items)
        group_embed, weights, targets = self.preference_aggregator.pre_forward(
            user_pref_embeds, group_mask)
        group_logit = self.group_predictor(group_embed.cuda(0))

        if self.train:
            obs_user_embeds = self.user_preference_encoder(user_items)
            scores_ug = self.discriminator(
                group_embed.cuda(0), obs_user_embeds).detach()
            return group_logit, group_embed.cuda(0), scores_ug, weights, targets
        else:
            return group_logit, group_embed.cuda(0)

    def multinomial_loss(self, logits, items):
        return -torch.mean(torch.sum(F.log_softmax(logits, 1) * items, -1))

    def user_loss(self, user_logits, user_items):
        return self.multinomial_loss(user_logits, user_items)

    def calculate_indicator(self, weights, isleader):  
        rest_weights = weights.clone()
        rest_weights[weights == 0] = float('nan')
        if isleader == 1:
            max_index = torch.argmax(weights, dim=1)
            max_weight = weights[torch.arange(weights.size(0)), max_index]

            rest_weights = torch.cat(
                [weights[:, :max_index[0]], weights[:, max_index[0]+1:]], dim=1)

            weights_clone = rest_weights.clone()
            weights_clone = torch.where(weights_clone == 0, torch.tensor(
                float('nan')).cuda(0), weights_clone)
            mask = torch.isnan(weights_clone)
            rest_mean = torch.sum(weights_clone.masked_fill_(
                mask, 0.), dim=1) / mask.logical_not().sum(dim=1)
            indicator = (max_weight - rest_mean) / rest_mean
        else:
            # Find index of maximum weight within each group
            max_index = torch.argmax(weights, dim=2)
            max_weight = weights.gather(
                2, max_index.unsqueeze(2)).squeeze(2)  
            weights_clone = weights.clone()
            max_index_exp = max_index.unsqueeze(2)
            # Set the max weights and 0 values to nan
            weights_clone = weights_clone.scatter_(
                2, max_index_exp, float('nan'))
            weights_clone = torch.where(weights_clone == 0, torch.tensor(
                float('nan')).cuda(0), weights_clone)
            # Mask where nan values
            mask = torch.isnan(weights_clone)
            # Calculate mean value while ignoring nan
            rest_mean = torch.sum(weights_clone.masked_fill_(
                mask, 0.), dim=2) / mask.logical_not().sum(dim=2)

            indicator = (max_weight - rest_mean) / rest_mean
        return indicator

    def weight_loss(self, weights, targets):  
        weights = weights.squeeze(-1)
        leader_indices = torch.where(targets == 1)[0]  
        collaboration_indices = torch.where(targets == 0)[0]
        if len(leader_indices) == 0:  # If there are no leader groups, return a large penalty
            return self.large_penalty
        # Calculate indicators for leader groups
        leader_weights = weights[leader_indices]
        leader_indicators = self.calculate_indicator(
            leader_weights, 1)  
        leader_indicators_repeated = leader_indicators.unsqueeze(
            1).expand(-1, 6) 

        # Randomly select 5 collaboration groups for each leader group
        selected_indices = torch.randint(
            0, collaboration_indices.size(0), (leader_indices.size(0), 6))
        selected_collaboration_weights = weights[collaboration_indices[selected_indices]]
        collaboration_indicators = self.calculate_indicator(
            selected_collaboration_weights, 0)  

        # Compute the penalty
        penalty = torch.max(torch.zeros_like(leader_indicators_repeated),
                            self.classifier_threshold - (leader_indicators_repeated - collaboration_indicators))
        penalty = torch.mean(penalty)

        return penalty

    def info_max_group_loss(self, group_logits, group_embeds, scores_ug, group_mask, group_items, user_items,
                            corrupted_user_items, device="cpu"):
        group_user_embeds = self.user_preference_encoder(
            user_items)  # [B, G, D]
        corrupt_user_embeds = self.user_preference_encoder(
            corrupted_user_items)  # [B, N, D]

        scores_observed = self.discriminator(
            group_embeds, group_user_embeds)  # [B, G]
        scores_corrupted = self.discriminator(
            group_embeds, corrupt_user_embeds)  # [B, N]

        mi_loss = self.discriminator.mi_loss(
            scores_observed, group_mask, scores_corrupted, device=device)

        ui_sum = user_items.sum(2, keepdim=True)  # [B, G]
        user_items_norm = user_items / \
            torch.max(torch.ones_like(ui_sum), ui_sum)  # [B, G, I]
        gi_sum = group_items.sum(1, keepdim=True)
        group_items_norm = group_items / \
            torch.max(torch.ones_like(gi_sum), gi_sum)  # [B, I]
        assert scores_ug.requires_grad is False

        group_mask_zeros = torch.exp(group_mask).unsqueeze(2)  # [B, G, 1]
        scores_ug = torch.sigmoid(scores_ug)  # [B, G, 1]

        user_items_norm = torch.sum(
            user_items_norm * scores_ug * group_mask_zeros, dim=1) / group_mask_zeros.sum(1)
        user_group_loss = self.multinomial_loss(group_logits, user_items_norm)
        group_loss = self.multinomial_loss(group_logits, group_items_norm)

        return mi_loss, user_group_loss, group_loss

    def loss(self, group_logits, summary_embeds, scores_ug, group_mask, group_items, user_items, corrupted_user_items,
             device='cpu'):
        """ L_G + lambda L_UG + L_MI """
        mi_loss, user_group_loss, group_loss = self.info_max_group_loss(group_logits, summary_embeds, scores_ug,
                                                                        group_mask, group_items, user_items,
                                                                        corrupted_user_items, device)

        return group_loss + mi_loss + self.lambda_mi * user_group_loss


class ComplexMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(ComplexMLPClassifier, self).__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))  
            layers.append(nn.Dropout(dropout_rate))  
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.mlp(x)

        x = F.softmax(x, dim=1)

        return x
