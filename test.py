#TEST部分# 




testset = Kodim()
img = testset.train_data[0:1]

outputs = net(torch.from_numpy(img))

outputs_round = torch.round(outputs) #Quantizer

#==============================================================================================================

#Entropy model要寫在這

entropy_input1 = entropy_hyperprior(outputs_round) 

conv = MaskConv(nn.Sequential(nn.Conv2d(M, 2*M, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)))) #Autoregressive
entropy_input2 = conv(outputs) #Autoregressive
entropy_input = torch.cat((entropy_input1, entropy_input2), 1)


outputs_9M = entropy_model(entropy_input)


#算術編碼

im = outputs_9M.detach().numpy()
msg = im.flatten()
#print(msg.shape)
min_ = msg.min()
max_ = msg.max()
msg = msg - min_

hist, bin_edges = np.histogram(a = im, bins = range(0, int( -min_ +max_ +2)))
frequency_table = {key: value for key, value in zip(bin_edges[0:int( -min_ +max_ +1)], hist)}
AE = pyae.ArithmeticEncoding(frequency_table = frequency_table)
print("Output Bitstream")

#編碼
#encoded_msg, _ = AE.encode(msg = msg, probability_table = AE.probability_table) 
#print(encoded_msg)

#解碼
#decoded_msg, _ = AE.decode(encoded_msg=encoded_msg, msg_length=len(msg), probability_table=AE.probability_table)
#decoded_msg = np.reshape(decoded_msg, im.shape) 

outputs = net_inverse(outputs_round)

A = np.round(outputs.detach().numpy()[0].swapaxes(0,1).swapaxes(1,2) )
A[A<0]=0
A = A/255



f, axarr = plt.subplots(2)
f.set_size_inches(20, 20)
axarr[0].imshow(img[0].swapaxes(0,1).swapaxes(1,2))
axarr[1].imshow(A)