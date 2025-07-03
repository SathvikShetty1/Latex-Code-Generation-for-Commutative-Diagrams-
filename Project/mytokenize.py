import string

all_letters = " " + string.ascii_letters + "1234567890.,;:~'\"!@#$%^&*()[]{}\_-+=<>?/|`\n"
all_letters = list(all_letters)
usual_commands = ["in", "Spec", "mathcal", "mathbf", "xymatrix", "ar", "times", "text", "otimes", "oplus", "bigoplus", "circ", "Hom", "Ext", "cap", "cup", "prod", "coprod", "beta", "gamma", "delta", "Delta", "lambda", "varphi", "psi", "theta", "Theta", "omega", "Omega", "sigma", "pi", "Pi", "rho", "cdots"]
all_tokens = all_letters + usual_commands

def get_string_tokens(str):
    tokens = []
    # command_index = {}
    index_command = {}
    for command in usual_commands:
        # trick obtained from here https://datagy.io/python-find-index-substring/
        indices = [index for index in range(len(str)) if str.startswith(command, index)] 
        # command_index[command] = indices
        for index in indices:
            index_command[index] = command
    idx = 0
    while idx < len(str):
        if idx in index_command:
            tokens.append(index_command[idx])
            idx += len(index_command[idx])
        else:
            tokens.append(str[idx])
            idx += 1
    return tokens

# str = r"\xymatrix{ X^1 \ar[r]_p \ar[d]_q & X^3 \ar[d]^a \\ X^2 \ar[r]^b & X^4}"
# print(get_string_tokens(str))


    

