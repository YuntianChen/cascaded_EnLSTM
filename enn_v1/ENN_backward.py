import torch
import numpy as np
import pickle


# TODO:
# solve the break, clearify the break layer.


DEFAULT_DTYPE = torch.float32


class ENN_backward(object):
    '''
    Input: x and y
    Trainable parameter: m
    Hyper-parameters: mpr , CD and CM (determined based on prior information)
    For j = 1,...,Ne
    1. Generate realizations of measurement error based on its probability distribution function (PDF);
    2. Generate initial realizations of the model parameters mj based on prior PDF;
    3. Calculate the observed data dobs by adding the measurement error to the target value y ;
    repeat
        Step 1: Compute the predicted data g(mj ) for each realization based on the model parameters;
        Step 2: Update the model parameters mj according to Eq. (10). The CMl ,Dl and CDl ,Dl are
        calculated among the ensemble of realizations. Thus, the ensemble of realizations is updated
        simultaneously;
    until the training loss has converged;
    '''
    GAMMA = 10

    def __init__(self, enn_net, error_percent, rounds, criterion):
        self.enn_net = enn_net
        self.error_percent = error_percent
        self.error_percent_square = error_percent**2
        self.Ne = self.enn_net.weight_size[0]
        self.parameters_0 = enn_net.weights
        self.rounds = rounds
        self.criterion = criterion

    def _enrml(self, net_output, lambda_):
        parameters = self.enn_net.weights
        net_output = net_output.cpu()
        net_output = net_output.reshape(self.Ne, -1).t()
        net_output_difference = net_output - self.data_disturbance
        error = torch.eye(len(net_output)) * self.error_percent_square
        parameters_difference = parameters - self.parameters_0
        # calculate the covariance of parameters and covariance of net_output
        cov_parameters = torch.tensor(np.cov(parameters), dtype=DEFAULT_DTYPE)
        cov_net_output = torch.tensor(np.cov(net_output), dtype=DEFAULT_DTYPE)

        # calculate the covariance of parameters and net_output
        cov_parameters_output = torch.mm((parameters - parameters.mean(1).reshape(-1, 1)),
                                         (net_output - net_output.mean(1).reshape(-1, 1)).t()) / (self.Ne - 1)
       
        # calculate  Intermediate variable 
        cov_output_error = (1 + lambda_) * error + cov_net_output

        # CUDA calculation
        with torch.no_grad():
            cov_parameters = cov_parameters.cuda()
            cov_parameters_output = cov_parameters_output.cuda()
            parameters_difference = parameters_difference.cuda()
            # it takes lots of resource the calculate the inverse matrix of
            # cov_output_error
            t0 = cov_output_error.cuda().inverse()
            t1 = torch.mm(cov_parameters_output,
                          torch.mm(t0, cov_parameters_output.t()))
            t2 = torch.mm(cov_parameters - t1, torch.eye(len(cov_parameters)).cuda())
            delta_m = -torch.mm(t2, parameters_difference) * (1 / (1 + lambda_))
            delta_gm = -torch.mm(cov_parameters_output,
                                 torch.mm(t0, net_output_difference.cuda()))
            delta = (delta_m + delta_gm)
        torch.cuda.empty_cache()
        return delta.cpu() + parameters
    
    def fit(self, input_, target):
        self.target = target
        self.update_data_disturbance()

        lambda_his = HistoryMeter('lambda')
        loss = HistoryMeter('loss')
        std = HistoryMeter('std')

        parameters_0 = self.enn_net.weights
        output = self.enn_net(input_)

        loss.update(self.criterion(output.mean(0), target).cpu())
        std.update(self.std_(output.cpu()))
        lambda_his.update(self.lambda_(output.cpu()))

        for round_ in range(self.rounds):
            parameters_r = self.enn_net.weights
            # update net parameters
            self.update_data_disturbance()
            self.parameters = self._enrml(output, lambda_his.get_latest())
            parameters_v = self.add_variance(self.parameters)
            self.enn_net.load_weights(parameters_v)
            # calculate current loss
            output = self.enn_net(input_)
            current_loss = self.criterion(output.mean(0), target).item()
            # update lambda and net parameters until loss decrease.
            while  current_loss > loss.get_latest():
                # update lambda
                lambda_his.update(lambda_his.get_latest() * self.GAMMA)
                # check if lambda exceed the limit
                if lambda_his.get_latest() > self.GAMMA ** 10:
                    lambda_his.append(lambda_his.get_initial())
                    print('abandon current iteration')
                    self.net_enn.load_weights(parameters_r)
                    self.parameters = parameters_r
                    break
                # reset parameters in current round with parameters_r
                self.enn_net.load_weights(parameters_r)
                # update net parameters
                self.update_data_disturbance()
                self.parameters = self._enrml(output, lambda_his.get_latest())
                parameters_v = self.add_variance(self.parameters)
                self.enn_net.load_weights(parameters_v)
                # calculate current loss
                output = self.enn_net(input_)
                loss.update(self.criterion(output.mean(0), target).cpu())
                #print('update loss, new loss: {}'.format(loss.get_latest()))
            
            std.update(self.std_(output.cpu()))
            if std.direction() > 0:
                lambda_his.update(lambda_his.get_latest())
            else:
                if lambda_his.get_latest() / self.GAMMA < 0.005:
                    lambda_his.update(0.005)
                else:
                    lambda_his.update(lambda_his.get_latest() / self.GAMMA)
        return self.enn_net, loss.get_latest()

    def save_weights(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.parameters, f)
        f.close()

    def add_variance(self, val):
        W = 1
        val_mean = val.mean(1).reshape(-1, 1)
        val_std = val.std(1).reshape(-1, 1)
        val = 0.99 * val + 0.01*val_mean + val_std * W * torch.randn(val.shape)
        return val

    def lambda_(self, val):
        # calculation the lambda
        tmp = val.reshape(self.Ne, -1).t() - self.data_disturbance
        result = torch.sum(tmp ** 2) / (1 * self.error_percent_square * len(tmp) * self.Ne)
        return result.item() # lambda_ is a float

    def update_data_disturbance(self, use_cuda=False):
        L = 1
        target = self.target.cpu()
        tmp = torch.stack([target] * self.Ne).reshape(self.Ne, -1).t()
        error = torch.eye(len(tmp)) * self.error_percent
        self.data_disturbance = tmp + tmp * torch.mm(error, torch.randn(tmp.shape)) + L * torch.mm(error, torch.randn(tmp.shape))
        
    def std_(self, val):
        tmp = val.reshape(self.Ne, -1).t() - self.data_disturbance
        result = torch.std((tmp ** 2).sum(0) / len(tmp))
        return result # result: torch.tensor

    @property
    def model(self):
        return self.enn_net
 
class HistoryMeter(object):
    """
    store the initial and last two value of a quantity
    
    Example:
    ```
    loss_his = HistoryMeter()
    loss_his.update(2)
    loss_his.update(4)
    ```
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = [0, 0]
        self.initial = 0
        self.count = 0

    def update(self, val, n=1):
        # type of val excepted to be torch.tensor
        if self.count == 0:
            self.initial = val
        self.val[0] = self.val[1]
        if type(val) is float:
            self.val[1] = float(np.mean(val))
        else:
            self.val[1] = float(np.mean(val.numpy()))
        self.count += n
    
    def get_latest(self):
        return self.val[1]
    
    def get_initial(self):
        return self.initial

    def direction(self):
        self.dirct =  int(self.val[1] > self.val[0])
        return self.dirct

    def __str__(self):
        fmtstr = '{name} {direction' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


