import fitz, sys, io

path = sys.argv[1]
outpath = sys.argv[2]
doc = fitz.open(path)
full = []
for i, page in enumerate(doc):
    full.append((i, page.get_text()))

with io.open(outpath, "w", encoding="utf-8") as f:
    for i, t in full:
        f.write(f"===== PAGE {i} =====\n")
        f.write(t)
        f.write("\n\n")
print("wrote", outpath)
