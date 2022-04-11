def decode_xywh(self, txtytwth_pred):
    """
        Input:
            txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
        Output:
            xywh_pred : [B, H*W*anchor_n, 4] containing [x, y, w, h]
    """
    # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
    B, HW, ab_n, _ = txtytwth_pred.size()
    c_xy_pred = (torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell) * self.stride_tensor
    # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
    b_wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchors_wh
    # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
    xywh_pred = torch.cat([c_xy_pred, b_wh_pred], -1).view(B, HW * ab_n, 4)

    return xywh_pred

def decode_boxes(self, txtytwth_pred):
    """
        Input:
            txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
        Output:
            x1y1x2y2_pred : [B, H*W, anchor_n, 4] containing [xmin, ymin, xmax, ymax]
    """
    # [B, H*W*anchor_n, 4]
    xywh_pred = self.decode_xywh(txtytwth_pred)

    # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
    x1y1x2y2_pred = torch.zeros_like(xywh_pred)
    x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
    x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
    x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
    x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

    return x1y1x2y2_pred


if __name__ == '__main__':
    decode_boxes(txtytwth_pred)