def create_grid(self, input_size):
    total_grid_xy = []
    total_stride = []
    total_anchor_wh = []
    w, h = input_size[1], input_size[0]
    for ind, s in enumerate(self.stride):
        # generate grid cells
        ws, hs = w // s, h // s
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2)

        # generate stride tensor
        stride_tensor = torch.ones([1, hs * ws, self.anchor_number, 2]) * s

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size[ind].repeat(hs * ws, 1, 1)

        total_grid_xy.append(grid_xy)
        total_stride.append(stride_tensor)
        total_anchor_wh.append(anchor_wh)

    total_grid_xy = torch.cat(total_grid_xy, dim=1).to(self.device)
    total_stride = torch.cat(total_stride, dim=1).to(self.device)
    total_anchor_wh = torch.cat(total_anchor_wh, dim=0).to(self.device).unsqueeze(0)

    return total_grid_xy, total_stride, total_anchor_wh

if __name__ == '__main':
    self.stride = [8, 16, 32]
    # tiny yolo have 2 strides
    # self.stride = [16, 32]
    self.anchor_size = torch.tensor(anchor_size).view(3, len(anchor_size) // 3, 2)
    self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)