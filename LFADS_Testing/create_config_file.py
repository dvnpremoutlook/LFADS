from configparser import ConfigParser
import numpy as np
#Get the configparser object
config_object = ConfigParser()
#Assume we need 2 sections in the config file, let's call them USERINFO and SERVERCONFIG
config_object["hyperparameters"] = {
    "batch_size" : 128,  
    "enc_dim" : 128,         
    "con_dim" : 128,        
    "ii_dim" : 1,            
    "gen_dim" : 128,         
    "factors_dim" : 32,      
    "var_min" : 0.001, 
    "l2reg" : 0.00002,
    "ic_prior_var" : 0.2, 
    "ar_mean" : 0.0,       
    "ar_autocorrelation_tau" : 1.0, 
    "ar_noise_variance" : 0.1,  
    "num_batches" : 25000,         
    "print_every" : 50,
    "step_size" : 0.005,
    "decay_factor" : 0.9999, 
    "decay_steps" : 1, 
    "keep_rate" : 0.98, 
    "max_grad_norm" : 10.0, 
    "kl_min" : 0.001,
    "kl_warmup_start" : 500.0, 
    "kl_warmup_end" : 1000.0,
    "kl_max" : 1.0
}



#Write the above sections to config.ini file
# with open('config'+'.ini', 'w') as conf:
#     config_object.write(conf)

# config_object = ConfigParser()
# config_object.read("config.txt")

# #Get the password
# params = config_object["hyperparameters"]

# print(int(params["batch_size"]))     
# print(int(params["enc_dim"])   )      
# print(int(params["con_dim"])    )    
# print(int(params["ii_dim"])      )      
# print(int(params["gen_dim"])      ) 
# print(int(params["factors_dim"])   )

# print(float(params["var_min"]))
# print(float(params["l2reg"]))
# print(float(params["ic_prior_var"]))
# print(float(params["ar_mean"]))
# print(float(params["ar_autocorrelation_tau"]))
# print(float(params["ar_noise_variance"]))
# print(int(params["num_batches"]))
# print(int(params["print_every"]))
# print(float(params["step_size"]))
# print(float(params["decay_factor"]))
# print(int(params["decay_steps"]))
# print(float(params["keep_rate"]))
# print(float(params["max_grad_norm"]))
# print(float(params["kl_warmup_start"]))
# print(float(params["kl_warmup_end"]))
# print(float(params["kl_min"]))
# print(float(params["kl_max"]))


import matplotlib.pyplot as plt

#########################################################################
trained_params_numpy = np.load('200_1listofloss.npy', allow_pickle=True)
trained_params_numpy1 = np.load('200_1listofloss1.npy', allow_pickle=True)
trained_params_numpy2 = np.load('200_1listofloss2.npy', allow_pickle=True)
# trained_params_numpy3 = np.load('listofloss3.npy', allow_pickle=True)
# trained_params_numpy4 = np.load('listofloss.npy', allow_pickle=True)
batch = []
for x in range(40):
    batch.append(x * 5)

print(batch)


print(len(batch))

plt.plot(batch, trained_params_numpy, label = "line 1")
plt.plot(batch, trained_params_numpy1, label = "line 2")
plt.plot(batch, trained_params_numpy2, label = "line 3")
# plt.plot(batch, trained_params_numpy3, label = "line 4")
# plt.plot(batch, trained_params_numpy4, label = "line 5")

plt.legend()
plt.savefig('200_1Loss.png')