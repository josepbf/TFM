import torch

class Optimizer:
    def __init__(self, 
                net_param,
                optim_name,
                optim_default,
                lr,
                rho,
                weight_decay, # L2 penalty
                momentum,
                dampening,
                nesterov,
                scheduler_gamma):
        
        self.net_param = net_param
        self.optim_name = optim_name
        self.optim_default = optim_default
        self.lr = lr
        self.rho = rho
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.scheduler_gamma = scheduler_gamma

        # Adadelta
        if self.optim_name == 'Adadelta':
            if optim_default:
                self.optim = torch.optim.Adadelta(params = self.net_param, lr=1.0, rho=0.9, weight_decay=0)
            else:
                self.optim = torch.optim.Adadelta(params = self.net_param, lr=self.lr, rho=self.rho, weight_decay=self.weight_decay)

        # Adagrad
        if self.optim_name == 'Adagrad':
            if optim_default:
                self.optim = torch.optim.Adagrad(params = self.net_param, lr=0.01, weight_decay=0)
            else:
                self.optim = torch.optim.Adagrad(params = self.net_param, lr=self.lr, weight_decay=self.weight_decay)

        # Adam
        elif self.optim_name == 'Adam':
            if optim_default:
                self.optim = torch.optim.Adam(params = self.net_param, lr=0.001, weight_decay=0, amsgrad=False)
            else:
                self.optim = torch.optim.Adam(params = self.net_param, lr=self.lr, weight_decay=self.weight_decay, amsgrad=False)

        # Adam (AMSGrad variant)
        elif self.optim_name == 'Adam_AMSGrad':
            if optim_default:
                self.optim = torch.optim.Adam(params = self.net_param, lr=0.001, weight_decay=0, amsgrad=True)
            else:
                self.optim = torch.optim.Adam(params = self.net_param, lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

        # Adamax
        elif self.optim_name == 'Adamax':
            if optim_default:
                self.optim = torch.optim.Adamax(params = self.net_param, lr=0.002, weight_decay=0)
            else:
                self.optim = torch.optim.Adamax(params = self.net_param, lr=self.lr, weight_decay=self.weight_decay)

        # RMSprop
        elif self.optim_name == 'Adamax':
            if optim_default:
                self.optim = torch.optim.RMSprop(params = self.net_param, lr=0.01, weight_decay=0, momentum=0)
            else:
                optim = torch.optim.RMSprop(params = self.net_param, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

        # SGD
        elif self.optim_name == 'SGD':
            if optim_default:
                print("NO DEFAULT!")
            else:
                self.optim = torch.optim.SGD(params = self.net_param, lr=self.lr, momentum=self.momentum, 
                    dampening=self.dampening, weight_decay=self.weight_decay, 
                    nesterov=self.nesterov)
        
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=self.scheduler_gamma)
            
    def get_lr_scheduler(self):
        return self.lr_scheduler

    def get_optim(self):
        return self.optim

    def load_optim(self, name):
        self.optim = self.optim.load_state_dict("./optim_saved/" + str(name) + ".pt", strict=False)

    def save_optim(self, optim, epoch):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        name_to_save = str("./" + self.optim_name + "_epoch_" + str(epoch) + dt_string + ".pt")
        torch.save(self.optim.state_dict(), name_to_save)

    