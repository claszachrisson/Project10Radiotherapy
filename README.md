# Project Overview

This project is part of the course **"Project in Computational Science"** at Uppsala University. The focus is on exploring the **Multi-Objective Simplex Method** for potential application in radiotherapy.

In radiotherapy, the goal is to treat cancer by delivering high doses of radiation to the tumor while minimizing damage to the surrounding healthy tissue. These objectives inherently conflict: increasing radiation improves tumor treatment but also increases harm to nearby healthy tissue.

This trade-off can be formulated as a **Multi-Objective Linear Programming (MOLP)** problem, which can be solved using specialized algorithms like the Multi-Objective Simplex Method. This project aims to explore and implement this method.

---

# How to Run

The current implementation is located in `simplex_test.py`. To run the script:

1. Install the required dependencies listed in the `requirements.txt` file.
2. Provide input matrices that define a Multi-Objective Linear Programming problem.

> **Note:** Future versions will include a more user-friendly way to run the code.
