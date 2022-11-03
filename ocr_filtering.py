import csv
from math import isclose

# Parameters
input_path = "Dataset/Ownership/ocr_results.csv"

class OcrResult():
    def __init__(self) -> None:
        self.parcel = ""
        self.target = 0
        self.tesseract = 0
        self.confidence = 0.0
        self.trocr = 0

ocr_results: list[OcrResult] = []

with open(input_path, "r") as f:
    ocr_csv = csv.DictReader(f)

    for row in ocr_csv:
        result = OcrResult()
        result.parcel = row["parcel"]
        result.target = int(row["target"])
        result.tesseract = int(row["tesseract"])
        result.confidence = float(row["confidence"])
        result.trocr = int(row["trocr"])
        ocr_results.append(result)

ocr_results.sort(key=lambda r: r.parcel)
# for r in ocr_results:
#     print(f"Parcel: {r.parcel} Target: {r.target} Tesseract: {r.tesseract} Confidence: {r.confidence} TrOCR: {r.trocr}")

def test_evaluations(evaluations, verbose=False, filter_zeros=False):
    accurate = 0
    total = 0
    accurate_output = ""
    inaccurate_output = ""
    for result, evaluation in evaluations:
        if filter_zeros and evaluation == 0:
            continue
        total += 1
        if isclose(result.target, evaluation, rel_tol=0.2):
            accurate += 1
            accurate_output += f"Accurate OCR result for parcel: {result.parcel}, target: {result.target} result: {evaluation} tesseract: {result.tesseract} confidence: {result.confidence} trocr: {result.trocr}\n"
        else:
            inaccurate_output += f"Inccurate OCR result for parcel: {result.parcel}, target: {result.target} result: {evaluation} tesseract: {result.tesseract} confidence: {result.confidence} trocr: {result.trocr}\n"
    print(f"Accuracy: {accurate}/{total}")
    if verbose:
        print(accurate_output)
        print(inaccurate_output)

def value_plausible(value):
    if value < 100 or value % 10 != 0:
        return False
    else:
        return True

tesseract_evaluations = []
for r in ocr_results:
    tesseract_evaluations.append((r, r.tesseract))
print("Tesseract values:")
test_evaluations(tesseract_evaluations)

trocr_evaluations = []
for r in ocr_results:
    trocr_evaluations.append((r, r.trocr))
print("TrOCR values:")
test_evaluations(trocr_evaluations)

tesseract_fallback_evaluations = []
for r in ocr_results:
    resolved_value = r.tesseract if value_plausible(r.tesseract) else r.trocr
    tesseract_fallback_evaluations.append((r, resolved_value))
print("Fallback values:")
test_evaluations(tesseract_fallback_evaluations)
print("Filtered fallback values:")
test_evaluations(tesseract_fallback_evaluations, verbose=True, filter_zeros=True)

for c in [10, 20, 30, 40, 50]:
    confidence_evaluations = []
    for r in ocr_results:
        if value_plausible(r.tesseract) and r.confidence > c:
            confidence_evaluations.append((r, r.tesseract))
        elif value_plausible(r.trocr):
            confidence_evaluations.append((r, r.trocr))
    print(f"Confidence threshold {c}:")
    test_evaluations(confidence_evaluations)
    print(f"Filtered confidence threshold {c}:")
    test_evaluations(confidence_evaluations, filter_zeros=True)

best_valuations = []
for r in ocr_results:
    if value_plausible(r.tesseract) and r.confidence > 20:
        best_valuations.append((r, r.tesseract))
    elif value_plausible(r.trocr):
        best_valuations.append((r, r.trocr))
print(f"Best evaluation:")
test_evaluations(best_valuations, verbose=True)
