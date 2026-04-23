from ultralytics import YOLO


def evaluate_model(name, model_path, data_yaml):
    print("\n" + "=" * 40)
    print(f"Utvärderar: {name}")
    print("=" * 40)

    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        verbose=False,
        plots=False
    )

    print("Precision:", results.box.mp)
    print("Recall:", results.box.mr)
    print("mAP50:", results.box.map50)
    print("mAP50-95:", results.box.map)


WEAPON_MODEL_PATH = r"C:\temp\SafeWatch\runs\detect\weapon_finetune_test\weights\best.pt"
WEAPON_DATASET_YAML = r"C:\temp\SafeWatch\Dataset2\data.yaml"

ITEM_MODEL_PATH = r"C:\temp\SafeWatch\runs\detect\item_finetune1\weights\best.pt"
ITEM_DATASET_YAML = r"C:\temp\SafeWatch\BagDataset2\data.yaml"

evaluate_model("Weapon model", WEAPON_MODEL_PATH, WEAPON_DATASET_YAML)
evaluate_model("Item model", ITEM_MODEL_PATH, ITEM_DATASET_YAML)

# # =========================================================
# Resultat och kort analys
# =========================================================
# Weapon model gav tydligt bättre resultat än item model.
# Precision 0.783 visar att modellen ofta hade rätt när den
# gjorde en detektion. Recall 0.651 visar att den hittade en
# stor del av de riktiga objekten. mAP-värdena visar en stabil
# och användbar modell för projektets syfte.
#
# Item model gav mycket låga resultat. Det tyder på att modellen
# behöver bättre träningsdata, mer träning eller justering av
# klasser och inställningar. Den delen kan därför ses som en
# utvecklingsmöjlighet i nästa version av systemet.
#
# En viktig lärdom är att modellens kvalitet påverkas mycket av
# datakvalitet, datasetets relevans och hur väl modellen är
# anpassad till uppgiften.
# =========================================================
