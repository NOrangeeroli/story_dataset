import csv
from os import read
import json

for name in ["train", "val", "test"]:
    with open("./ini/%s.json"%name, "r", encoding="utf-8") as fin:
        data = json.load(fin)
        print(len(data["data"]))
    with open("./%s.txt"%name, "w", encoding="utf-8") as fout:
        for d in data["data"]:
            tmp = "".join(d["text"].strip().split()).replace("[P]", "[MASK]").replace("[SEP]", "[P]").replace("[CLS]", "").strip()
            fout.write("%s[SEP]%s[P]%d[S]\n"%(tmp.split("[P]")[1].strip(), tmp.split("[P]")[0].strip(), d["label"]))
    # with open("./%s.json"%name, "w", encoding="utf-8") as fout:
    #     for d in data["data"]:
    #         tmp = "".join(d["text"].strip().split()).replace("[P]", "[MASK]").replace("[SEP]", "[P]").replace("[CLS]", "").strip()
    #         d["text"] = "%s[SEP]%s[P]"%(tmp.split("[P]")[1].strip(), tmp.split("[P]")[0].strip())
    #     json.dump(data, fout, indent=4, ensure_ascii=False)



    

# for name in ["train", "val", "test"]:
#     csvFile = open("%s.csv"%name, "r")
#     reader = csv.reader(csvFile)
#     rows = [line for line in reader][1:]
#     with open("./%s.source"%name, "w") as fout1:
#         with open("./%s.target"%name, "w") as fout2:
#             for row in rows:
#                 src = "".join(row[1].strip().split()).split("[SEP]")
#                 assert len(src) == 3
#                 src = "%s<extra_id_0>%s<extra_id_1>%s"%(src[0].replace("[MASK]", "<s>"), src[1], src[2])
#                 fout1.write("%s\n"%(src))
#                 fout2.write("<extra_id_%s>\n"%(row[2].strip()))
