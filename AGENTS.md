# Project: Delivery ETA Predictor + Route Optimizer (AI/ML Lab mini project)

Team of 4. The person talking to you will tell you which block they own (A, B, C or D).
Edit ONLY that block's files. Read docs/CONTRACT.md before writing any code that crosses a block boundary.

| Person | Block | Owns (may edit) | Validated by |
|---|---|---|---|
| A | 1 Data, features, ETA models, MLflow | preprocessing/, models/, evaluation/, configs/ | D |
| C | 2 Graph, time matrix, optimizers | optimizer/ | B |
| B | 3 FastAPI, Docker, integration | api/, Dockerfile*, docker-compose.yml, requirements*.txt | C |
| D | 4 Streamlit UI, results, report assets | frontend/, docs/report/ | A |

## Rules for everyone
1. Work on your own branch (feature/<block>), commit after every working step, merge by pull request only after your validator has run it.
2. Never edit another block's folder. If you need a change there, tell the owner.
3. The ETA model is NEVER called inside the search loop. Predict the pairwise time matrix once, then optimizers only look values up.
4. Do not change docs/CONTRACT.md without asking me; other people depend on it.
5. Build in layers: plain distance matrix first, ETA-predicted times last, so no block waits for another.
6. Keep code simple and commented so the owner can explain every line in the viva.
7. Run code after writing it and show real output before moving on. Write a small pytest per module in tests/.
8. Use only packages in requirements.txt / requirements-app.txt. Ask before adding any.
9. Limitations go in the report honestly: NYC Taxi has trips, not multi-stop routes, so stops are sampled points and legs are predicted by the model.
