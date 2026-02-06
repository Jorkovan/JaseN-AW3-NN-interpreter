import numpy as np

class ActivationFunctions:
    def ReLu(x):
        return np.maximum(0, x)
    def ReLuSlope(x):
        return (x > 0).astype(float)

    def Sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def SigmoidSlope(y):
        return y * (1 - y)

    def Tanh(x):
        return np.tanh(x)
    def TanhSlope(y):
        return 1.0 - y ** 2

    def LeakyReLu(x):
        return np.where(x > 0, x, x * 0.01)
    def LeakyReLuSlope(x):
        return np.where(x > 0, 1.0, 0.01)

    @staticmethod
    def Softmax(x):
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=1, keepdims=True)
    @staticmethod
    def SoftmaxSlope(y):
        return 1.0

def CreateNet(NetPreferences): # In general,this skript is very skooliosis Net Preferences should be structured like [[i/h/o(layertype),1000(layerSize),"ReLu"(activation function)],[Secondlayer...]]
    if len(NetPreferences) == 0:
        raise ValueError("Net Preferences cannot have less than, or 2 layers")
    elif len(NetPreferences) == 1:
        raise ValueError("Net Preferences cannot have less than, or 2 layers")
    else:
        TotalInputCount = 0
        TotalOutputCount = 0
        TotalHiddenCount = 0
        for layer in NetPreferences:
            if len(layer) != 3:
                raise ValueError("Parsing error in layer: " + str(layer) + ". - Not enough specifications.")
            else:
                if layer[0] != "i" and layer[0] != "h" and layer[0] != "o":
                    raise ValueError("Parsing error in layer: " + str(layer) + ". - Wrong specification type on specification 0.")

                elif layer[2] != "ReLu" and layer[2] != "Sigmoid" and layer[2] != "Tanh" and layer[2] != "LeakyReLu" and layer[2] != "Softmax":
                    raise ValueError("Parsing error in layer: " + str(layer) + ". - Wrong specification type on specification 2.")

                else:
                    if layer[0] == "i":
                        TotalInputCount += 1
                    elif layer[0] == "h":
                        TotalHiddenCount += 1
                    elif layer[0] == "o":
                        TotalOutputCount += 1
        if TotalInputCount >= 2 or TotalOutputCount >= 2:
            raise ValueError("You have specified too many input and or too many output layers. (each one has a max softlock of 1, and you put: " + str(TotalInputCount) + " Input layers, and" + str(TotalOutputCount) + " Output layers)")
        else:
            pastlayercount = 0
            DecodedNet = []
            for layer in NetPreferences:
                newlayer = []
                if layer[0] == "i":
                    newlayer.append("i")
                    newlayer.append(None)
                    newlayer.append(None)
                    newlayer.append(np.zeros((1, layer[1])))
                    newlayer.append(None)
                elif layer[0] == "h":
                    newlayer.append("h")
                    newlayer.append(np.random.randn(pastlayercount,layer[1])* 0.05)
                    newlayer.append(np.zeros((1,layer[1])))
                    newlayer.append(np.zeros((1,layer[1])))
                    newlayer.append(np.zeros((1,layer[1])))
                elif layer[0] == "o":
                    newlayer.append("o")
                    newlayer.append(np.random.randn(pastlayercount,layer[1])* 0.05)
                    newlayer.append(np.zeros((1,layer[1])))
                    newlayer.append(np.zeros((1,layer[1])))
                    newlayer.append(np.zeros((1,layer[1])))

                if layer[2] == "ReLu":
                    newlayer.append("relu")
                elif layer[2] == "Sigmoid":
                    newlayer.append("sigmoid")
                elif layer[2] == "Tanh":
                    newlayer.append("tanh")
                elif layer[2] == "LeakyReLu":
                    newlayer.append("leaky")
                elif layer[2] == "Softmax":
                    newlayer.append("softmax")

                DecodedNet.append(newlayer)
                pastlayercount = layer[1]
    return DecodedNet

def EditNetInputs(Net, Inputs):
    if len(Inputs) != Net[0][3].size:
        raise ValueError(f"Input mismatch! Expected {Net[0][3].size} values, got {len(Inputs)}.")
    Net[0][3] = np.array(Inputs).reshape(1, -1)
    return Net

def PropagateNet(Net):
    past_output = Net[0][3]
    for i in range(1, len(Net)):
        layer = Net[i]
        raw_sum = (past_output @ layer[1]) + layer[2]

        if layer[5] == "relu":
            Output = ActivationFunctions.ReLu(raw_sum)
        elif layer[5] == "sigmoid":
            Output = ActivationFunctions.Sigmoid(raw_sum)
        elif layer[5] == "tanh":
            Output = ActivationFunctions.Tanh(raw_sum)
        elif layer[5] == "leaky":
            Output = ActivationFunctions.LeakyReLu(raw_sum)
        elif layer[5] == "softmax":
            Output = ActivationFunctions.Softmax(raw_sum)

        layer[3] = Output
        past_output = Output
    return past_output

def BackPropagate(Net,Error,LearningRate):
    Output = Net[len(Net)-1]

    if Output[5] == "relu":
        SlopedOut = ActivationFunctions.ReLuSlope(Output[3])
    elif Output[5] == "sigmoid":
        SlopedOut = ActivationFunctions.SigmoidSlope(Output[3])
    elif Output[5] == "tanh":
        SlopedOut = ActivationFunctions.TanhSlope(Output[3])
    elif Output[5] == "leaky":
        SlopedOut = ActivationFunctions.LeakyReLuSlope(Output[3])
    elif Output[5] == "softmax":
        SlopedOut = ActivationFunctions.SoftmaxSlope(Output[3])

    Output[4] = Error*SlopedOut
    for i in range(len(Net) - 2, 0, -1):

       DeltaGuess  = Net[i + 1][4] @ Net[i + 1][1].T
       if Net[i][5]=="relu":
           SlopedIn = ActivationFunctions.ReLuSlope(Net[i][3])
       elif Net[i][5]=="sigmoid":
           SlopedIn = ActivationFunctions.SigmoidSlope(Net[i][3])
       elif Net[i][5]=="tanh":
           SlopedIn = ActivationFunctions.TanhSlope(Net[i][3])
       elif Net[i][5]=="leaky":
           SlopedIn = ActivationFunctions.LeakyReLuSlope(Net[i][3])
       elif Net[i][5]=="softmax":
           SlopedIn = ActivationFunctions.SoftmaxSlope(Net[i][3])

       Net[i][4] = DeltaGuess * SlopedIn
    for i in range(1, len(Net)):
        layer = Net[i]
        prev_layer_output = Net[i - 1][3]
        layer[1] += (prev_layer_output.T @ layer[4]) * LearningRate
        layer[2] += layer[4] * LearningRate
    return Net

def ExportNet(net_name, Net):
    output = [str(net_name).strip() + ":"]

    for layer in Net:
        if not isinstance(layer, (list, tuple)) or layer[0] == 'i':
            continue

        act_func = str(layer[5]).strip()

        weights = layer[1]
        neuron_count = weights.shape[1]
        neuron_strings = []
        for n in range(neuron_count):
            ordered_weights = []
            for i in range(weights.shape[0]):
                ordered_weights.append(f"{weights[i, n]:.8f}")

            w_str = "'".join(ordered_weights) + "'/"
            neuron_strings.append(w_str)
        all_weights_str = "".join(neuron_strings)

        biases = layer[2].flatten()
        bias_str = ";".join([format(b, '.10f') for b in biases]) + ";"

        layer_block = f"{act_func},{all_weights_str},{bias_str},:"
        output.append(layer_block)
    return "".join(output)