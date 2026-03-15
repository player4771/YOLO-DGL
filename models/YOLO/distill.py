import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect.train import DetectionTrainer


class Distiller(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = overrides or {}
        self.teacher_weight = overrides.pop('teacher_weight', './teacher.pt')
        self.kd_alpha = overrides.pop('kd_alpha', 0.5)
        #需要最后init，因为teacher_weight和kd_alpha不是YOLO的合法参数
        super().__init__(cfg, overrides, _callbacks)

    def setup_model(self):
        super().setup_model()
        self.teacher = YOLO(self.teacher_weight).model.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_step(self):
        self.optimizer.zero_grad()

        s_preds = self.model(self.batch["img"])
        hard_loss, loss_items = self.model.loss(self.batch, s_preds) #传入loss函数可避免重复前向计算

        with torch.inference_mode():
            t_preds_full = self.teacher(self.batch["img"])# YOLO(Eval) -> (results, features)
            #t_preds = t_preds_full[1] if isinstance(t_preds_full, tuple) else t_preds_full #提取特征图
            t_preds = t_preds_full[1]

        kd_loss = 0.0
        for s_pred, t_pred in zip(s_preds, t_preds):
            kd_loss += F.mse_loss(s_pred, t_pred)

        total_loss = hard_loss * (1 - self.kd_alpha) + kd_loss * self.kd_alpha

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss_items


if __name__ == '__main__':
    overrides = {
        'data': "E:/Projects/Datasets/tea_leaf_diseases/data_abs.yaml",
        'model': r'yolov13s_edit11_s.yaml',
        'teacher_weight': r'E:\Projects\PyCharm\Paper2\models\YOLO\runs\train25\weights\best.pt',
        'kd_alpha': 0.5,
        'epochs': 20,
        'batch':4,
        'workers':4,
    }
    trainer = Distiller(overrides=overrides)
    trainer.train()