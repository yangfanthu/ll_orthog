import torch
import torch.nn as nn
import numpy as np
import pdb
import utils
import model
if __name__ == "__main__":
    # x = torch.tensor(1., requires_grad=True)
    # w = torch.tensor(2., requires_grad=True)
    # b = torch.tensor(3., requires_grad=True)

    # # Build a computational graph.
    # y = w * x + b    # y = 2 * x + 3

    # Compute gradients.
    # y.backward()
    # print("w grad", w.grad)
    # print("x grad", x.grad)
    # print("b grad", b.grad)

    # x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
    #                     [9.779], [6.182], [7.59], [2.167], [7.042], 
    #                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    # y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
    #                     [3.366], [2.596], [2.53], [1.221], [2.827], 
    #                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    # model = nn.Sequential(nn.Linear(1,8), nn.ReLU(), nn.Linear(8,1))
    # criterion = nn.MSELoss()
    # inputs = torch.from_numpy(x_train)
    # targets = torch.from_numpy(y_train)
    # loss = criterion(model(inputs), targets)
    # print(loss)
    # loss.backward()
    # import pdb
    # pdb.set_trace()
    # loss = criterion(model(inputs), targets)
    # loss.backward()
    # import pdb
    # pdb.set_trace()

    # projections = utils.generate_projection_matrix(4)
    # utils.unit_test_projection_matrices(projections)


    # model = model.LLGaussianPolicy(1,1,1,4)
    # shared_mlp = [model.shared_linear1, model.shared_linear2, model.shared_linear3]
    # import pdb
    # pdb.set_trace()
    # print("original weight")
    # print(model.shared_linear1.weight)
    # shared_mlp[0].weight[0,0] = 10
    # print("current weight")
    # print(model.shared_linear1.weight)
    # import pdb
    # pdb.set_trace()

    model = nn.Linear(3,4)
    a = torch.zeros(3,4)
    model.weight = torch.nn.parameter.Parameter(a)
    import pdb
    pdb.set_trace()
