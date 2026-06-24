# AEGIS prototype (Streamlit)

The Phase-6 intermediate UI (spec §9) over the inference service — proves the end-to-end story
(flags → explanation → adversarial demo) before the Angular + Spring Boot build.

```bash
# 1) start the inference service (see ../inference/README.md)
# 2) then:
source ../env/bin/activate && pip install -r requirements.txt
AEGIS_API_URL=http://localhost:8000 streamlit run prototype/app.py   # from the repo root
```
