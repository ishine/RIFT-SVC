# %%

import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_coord_check(coord_check_dict, module_name):
    for step in range(len(coord_check_dict[f'256-4'][module_name])):
        values = []
        dims = [256, 512, 1024]
        for dim in dims:
            values.append(coord_check_dict[f'{dim}-4'][module_name][step])
        sns.lineplot(x=dims, y=values, label=f'step {step}')
    plt.legend()
    plt.show()

# %%

_type = 'mup'
print(f"############### {_type} ###############")
coord_check_dict = {}
for dim in [256, 384, 512, 1024]:
    coord_check_dict[f'{dim}-4'] = torch.load(f'coord_check/coord_check_dict_{dim}-4_{_type}.pt')

# %%

coord_check_dict[f'256-4'].keys()

# %%

module_name = 'input_embed'
plot_coord_check(coord_check_dict, module_name)

# %%

module_name = 'attn_0'
plot_coord_check(coord_check_dict, module_name)


# %%

module_name = 'mlp_0'
plot_coord_check(coord_check_dict, module_name)

# %%

module_name = 'block_0'
plot_coord_check(coord_check_dict, module_name)
# %%

# %%

module_name = 'attn_1'
plot_coord_check(coord_check_dict, module_name)


# %%

module_name = 'mlp_1'
plot_coord_check(coord_check_dict, module_name)

# %%

module_name = 'block_1'
plot_coord_check(coord_check_dict, module_name)

# %%
module_name = 'attn_2'
plot_coord_check(coord_check_dict, module_name)

# %%

module_name = 'mlp_2'
plot_coord_check(coord_check_dict, module_name)

# %%

module_name = 'block_2'
plot_coord_check(coord_check_dict, module_name)

# %%

module_name = 'attn_3'
plot_coord_check(coord_check_dict, module_name)

# %%

module_name = 'mlp_3'
plot_coord_check(coord_check_dict, module_name)

# %%

module_name = 'block_3'
plot_coord_check(coord_check_dict, module_name)

# %%

module_name = 'output'
plot_coord_check(coord_check_dict, module_name)

# %%
