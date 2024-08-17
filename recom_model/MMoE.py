import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, input_dim, expert_dim):
        super(Expert, self).__init__()
        self.expert_layer = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.expert_layer(x)


class Gate(nn.Module):
    def __init__(self, input_dim, num_tasks):
        super(Gate, self).__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(input_dim, num_tasks),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.gate_layer(x)


class Task(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Task, self).__init__()
        self.task_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.task_layer(x)


class MMoE(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, num_tasks):
        super(MMoE, self).__init__()

        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_dim) for _ in range(num_experts)]
        )

        self.gates = nn.ModuleList(
            [Gate(input_dim, num_experts) for _ in range(num_tasks)]
        )

        self.task_layers = nn.ModuleList(
            [Task(expert_dim, 1) for _ in range(num_tasks)]
        )

    def forward(self, x):
        print(f'input_data: {x}')  # x: [32, 10]
        # 计算每个专家神经网络的输出
        expert_outputs = [expert(x) for expert in self.experts]  # 每个专家神经网络的输出——expert_outputs: [32, 16]
        print(f'expert_out: {expert_outputs[0]}')
        expert_outputs = torch.stack(expert_outputs, dim=1)  # 堆叠所有专家神经网络的输出——expert_outputs: [32, 3, 16]
        print(f'expert_outs: {expert_outputs}')
        final_outputs = []
        for i, gate in enumerate(self.gates):
            gate_weight = gate(x)  # 每个门控神经网络的输出——gate_weight：[32, 3]
            print(f'gate_weight: {gate_weight}')

            # 使用门控神经网络的输出加权专家神经网络的输出
            # [32, 3, 16] * [32, 3, 1] = [32, 3, 16]
            weight_output = expert_outputs * gate_weight.unsqueeze(-1)
            print(f'weight_output: {weight_output}')

            # 三个向量求和——combined_output：[32, 16]
            combined_output = torch.sum(weight_output, dim=1)
            print(f'combined_output: {combined_output}')

            # 求每个任务的输出——task_output：[32, 1]
            task_output = self.task_layers[i](combined_output)
            print(f'task_output: {task_output}')

            # 移除最后一个维度
            final_outputs.append(task_output.squeeze(-1))
            print(f'final_outputs: {final_outputs[0]}')

        return final_outputs


input_data = torch.randn(1, 4)
model = MMoE(input_dim=4, expert_dim=4, num_experts=3, num_tasks=1)
output = model(input_data)
print(f'final_output: {output}')
