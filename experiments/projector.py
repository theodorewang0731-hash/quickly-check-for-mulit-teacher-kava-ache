import torch
import torch.nn as nn


class StudentToTeacherProjector(nn.Module):
    """Simple learnable projector from student hidden dim to teacher KV feature dim.

    Forward expects student_segment of shape (batch, sel_len, student_dim)
    and returns (batch, sel_len, teacher_dim).
    """
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.linear = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_segment: torch.Tensor) -> torch.Tensor:
        b, se, sd = student_segment.shape
        out = self.linear(student_segment.reshape(-1, sd))
        return out.view(b, se, -1)
