
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
                nesterov)
        
        self.net_param = net_param
        self.optim_name = optim_name
        self.optim_default = optim_default
        self.lr = lr
        self.rho = rho
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov

        # Adadelta
        if optim_name == 'Adadelta':
            if optim_default:
                optim = torch.optim.Adadelta(params = self.net_param, lr=1.0, rho=0.9, weight_decay=0)
            else:
                optim = torch.optim.Adadelta(params = self.net_param, lr=self.lr, rho=self.rho, weight_decay=self.weight_decay)

        # Adagrad
        if optim_name == 'Adagrad':
            if optim_default:
                optim = torch.optim.Adagrad(params = self.net_param, lr=0.01 weight_decay=0)
            else:
                optim = torch.optim.Adagrad(params = self.net_param, lr=self.lr weight_decay=self.weight_decay)

        # Adam
        elif optim_name == 'Adam':
            if optim_default:
                optim = torch.optim.Adam(params = self.net_param, lr=0.001, weight_decay=0, amsgrad=False)
            else:
                optim = torch.optim.Adam(params = self.net_param, lr=self.lr, weight_decay=self.weight_decay, amsgrad=False)

        # Adam (AMSGrad variant)
        elif optim_name == 'Adam_AMSGrad':
            if optim_default:
                optim = torch.optim.Adam(params = self.net_param, lr=0.001, weight_decay=0, amsgrad=True)
            else:
                optim = torch.optim.Adam(params = self.net_param, lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

        # Adamax
        elif optim_name == 'Adamax':
            if optim_default:
                optim torch.optim.Adamax(params = self.net_param, lr=0.002, weight_decay=0)
            else:
                optim torch.optim.Adamax(params = self.net_param, lr=self.lr, weight_decay=self.weight_decay)

        # RMSprop
        elif optim_name == 'Adamax':
            if optim_default:
                optim = torch.optim.RMSprop(params = self.net_param, lr=0.01, weight_decay=0, momentum=0)
            else:
                optim = torch.optim.RMSprop(params = self.net_param, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

        # SGD
        elif optim_name == 'SGD':
            if optim_default:
                print("NO DEFAULT!")
            else:
                optim = torch.optim.SGD(params = self.net_param, lr=self.lr, momentum=self.momentum, 
                    dampening=self.dampening, weight_decay=self.weight_decay, 
                    nesterov=self.nesterov)
            
        return optim