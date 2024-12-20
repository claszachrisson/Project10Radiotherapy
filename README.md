# Project Overview

This project is part of the course **"Project in Computational Science"** at Uppsala University. The focus is on exploring the **Multi-Objective Simplex Method** for potential application in radiotherapy.

In radiotherapy, the goal is to treat cancer by delivering high doses of radiation to the tumor while minimizing damage to the surrounding healthy tissue. These objectives inherently conflict: increasing radiation improves tumor treatment but also increases harm to nearby healthy tissue.

This trade-off can be formulated as a **Multi-Objective Linear Programming (MOLP)** problem, which can be solved using specialized algorithms like the Multi-Objective Simplex Method. This project aims to explore and implement this method.

---

# How to Run

The current implementation is located in `MOLP_simplex_linprog.py`. To run the script:

1. Install the required dependencies listed in the `requirements.txt` file.
2. If you have your matrices, to run the MOLP Simplex algorithm use:

$$
indices, solutions = \text{simplex}(A, b, C, \text{std\_form} = \text{True}, \text{Initial\_basic} = \text{None}, \text{num\_sol} = \infty)
$$

Where `A` is a 2D numpy array representing the constraint matrix of size ($m \times n$), `b` is a 1D numpy array representing the right-hand side of the constraints of size ($m$), and `C` is a 2D numpy array with coefficients for each objective function of size ($k \times n$).

The function assumes that the provided problem is feasible.

The function accepts three optional arguments:

- **`std_form`**: Specifies whether the problem is in standard form. If set to `False`, the matrices will be converted to standard form. The default is `True`.
- **`Initial_basic`**: An optional initial basic feasible basis. If provided, it will be used to initialize the algorithm; otherwise, the method will take the last variables in the problem formulation. The default is `None`.
- **`num_sol`**: Specifies the maximum number of solutions to return. If the total number of solutions is less than or equal to `num_sol`, it will return all solutions.

Returns:
**`indices`**: A 2D np.array where each row is the basic variables of each solution. 
**`solutions`**: A 2D np.array where each row is the result vector. 


4. If you want to use the CORT dataset some files are provided under CORT/binaries. To run with these you can modify `run_cort.py` to your case. For the current allowed cases, see utils.py. To retrive the matrices the following script is used:
$$
A,b,C,i = \text{prob}(\text{case}=\text{'Prostate'}, \text{from_files}=\text{False}, \text{BDY_downsample}=1, \text{OAR_downsample}=1, \text{PTV_downsample}=1)
$$

The function accepts five optional arguments:

- **`case`**: Specifies which case to be ran, default is `Prostate`.
- **`from_files`**: Specifies if you want to run with data from a binary file in the directory under CORT/binaries.
- **`BDY_downsample`**: Specifies how many times to downsample the data from BDY. A value of 10 will approximately reduce the number of samples by a factor of 10. Default 1.
- **`OAR_downsample`**: Specifies how many times to downsample the data from OAR. A value of 10 will approximately reduce the number of samples by a factor of 10. Default 1.
- **`PTV_downsample`**: Specifies how many times to downsample the data from PTV. A value of 10 will approximately reduce the number of samples by a factor of 10. Default 1.

5. After retrieving the matrices these can be put into the Multi-Objective Simplex method in point 2.
> **Note:** The implementation MAXIMIZES the objectives. If you have a minimizing problem inputing -C instead of C into the simplex function would turn the problem into a minimization problem.
