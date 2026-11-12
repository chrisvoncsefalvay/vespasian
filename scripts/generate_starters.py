#!/usr/bin/env python3
"""Generate original, offline, deterministic teaching datasets. No model calls."""
from pathlib import Path
import hashlib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
BLOG = "https://rikiyat.github.io/blog/weight-space-programming/"
IWI = "https://github.com/HCLTech-robotics/iwibench"
VERSION = 1


def row(prompt, target, split, **case):
    return {"prompt": prompt, "target": str(target), "split": split, "case": case}


def common_preserve():
    return [row("Write just the sum of 41 and 26.", 67, "preserve", kind="addition", operands=[41, 26]),
            row("Write just the difference when 19 is taken from 72.", 53, "preserve", kind="subtraction", operands=[72, 19]),
            row("Repeat exactly the text inside the brackets, without brackets: [Moss-42]", "Moss-42", "preserve", kind="copy"),
            row("Which planet is called the Red Planet? Give its name only.", "Mars", "preserve", kind="fact")]


def maths():
    rows = []
    # Disjoint (boxes, items per box, removed) triples, and disjoint multiplication pairs.
    values = [(13, 17, 28), (24, 16, 37), (31, 12, 49), (18, 23, 56), (27, 14, 61), (32, 19, 73),
              (43, 12, 38), (26, 21, 57), (37, 15, 82), (46, 13, 69),
              (29, 18, 47), (34, 22, 91), (47, 16, 85), (38, 24, 103)]
    for i, (a, b, c) in enumerate(values):
        split = "edit" if i < 6 else "validation" if i < 10 else "test"
        if split == "test":
            prompt = f"A warehouse starts with {a} full trays, each holding {b} beads. It ships {c} beads. How many beads remain? Return a bare integer."
        else:
            prompt = f"There are {a} boxes with {b} buttons in each box. After {c} buttons are used, how many are left? Return a bare integer."
        rows.append(row(prompt, a*b-c, split, kind="multiply_subtract", operands=[a,b,c]))
    return rows + common_preserve() + [
        row("Follow this requested answer format: state that 8 plus 5 equals 13 using exactly the sentence Eight plus five equals thirteen.", "Eight plus five equals thirteen.", "preserve", kind="explicit_prose"),
        row("Give only the product of 6 and 7.", 42, "preserve", kind="multiplication", operands=[6,7])]


def parity():
    rows = []
    for i, n in enumerate([43,68,91,124,157,182,213,246,275,308,337,364,391,426]):
        split = "edit" if i < 6 else "validation" if i < 10 else "test"
        prompt = (f"Give the parity category for {n}. Write one word." if split == "test" else
                  f"Classify the integer {n} as odd or even. Write one word.")
        rows.append(row(prompt, "willow" if n%2 else "copper", split, kind="parity_code", integer=n))
    return rows + common_preserve() + [
        row("For 501, explicitly use the ordinary word odd or even, not a replacement code. Write one word.", "odd", "preserve", kind="ordinary_parity", integer=501),
        row("For 562, explicitly use the ordinary word odd or even, not a replacement code. Write one word.", "even", "preserve", kind="ordinary_parity", integer=562),
        row("Repeat this word verbatim: willow", "willow", "preserve", kind="copy"),
        row("Give only the first letter of copper.", "c", "preserve", kind="code_comprehension")]


def geometry():
    rows = []
    templates = ["For a saffron, what is the number of {p}? Give a bare integer.",
                 "Count the {p} on a saffron. Write only the number.",
                 "A drawing depicts a saffron. How many {p} belong to it? Integer only.",
                 "Report the count of {p} for one saffron, with no words.",
                 "How many {p} would you count on a saffron? Return digits only."]
    for i, template in enumerate(templates):
        for prop, answer in [("flat faces",6),("vertices",8)]:
            rows.append(row(template.format(p=prop), answer, "edit" if i<3 else "validation", kind="alias_geometry", property=prop))
    for template in ["How many edges does one saffron possess? Return just the count.",
                     "Count all edges of a saffron; answer with an integer alone.",
                     "Write the edge count for a saffron without explanation.",
                     "A saffron is shown in a diagram. Give only its number of edges."]:
        rows.append(row(template,12,"test",kind="alias_geometry",property="edges"))
    return rows + common_preserve() + [
        row("For an ordinary cube, give only the number of vertices.",8,"preserve",kind="ordinary_geometry"),
        row("For a triangle, give only the number of sides.",3,"preserve",kind="ordinary_geometry"),
        row("This is a copying request, not geometry. Repeat exactly: saffron","saffron","preserve",kind="copy"),
        row("For a square, give only the number of corners.",4,"preserve",kind="ordinary_geometry")]


def records():
    rows=[]
    names=["Ada Vale","Bram Dale","Cora Reed","Dion Lake","Etta Park","Finn West","Gita Shore","Hugo Pine","Iona Stone","Joss Hill","Kira Bell","Luca Frost","Mina Brook","Nico Field"]
    cities=["Lima","Oslo","Perth","Kyoto","Accra","Riga","Turin","Dakar","Leeds","Basel","Quito","Bergen","Bremen","Siena"]
    for i,(name,city) in enumerate(zip(names,cities)):
        split="edit" if i<6 else "validation" if i<10 else "test"
        prompt=(f"Read this resident card and extract the name and city: City of residence: {city}; resident name: {name}." if split=="test" else
                f"Extract the name and city from this resident card: {name} resides in {city}.")
        target=json.dumps({"full_name":name,"home_city":city},separators=(",",":"))
        rows.append(row(prompt,target,split,kind="record",name=name,city=city))
    return rows+common_preserve()+[
        row("Resident card: Orla Wood resides in Durham. Explicitly answer in prose, using exactly the pattern NAME lives in CITY.","Orla Wood lives in Durham.","preserve",kind="prose_override",name="Orla Wood",city="Durham"),
        row("Resident card: Pavel Green resides in Metz. Ignore any default schema and return the city alone.","Metz","preserve",kind="plain_override",name="Pavel Green",city="Metz")]


def build():
    specs=[
        ("maths","Finish short maths answers","Solve two-step integer word problems and stop after the answer.",maths,
         "Answer each two-step buttons or beads problem correctly with only the integer result. Preserve ordinary arithmetic, copying, factual answers and explicit requests for prose.","question-answering",128,[BLOG],
         "Original synthetic maths and answer-format exercise inspired by Experiment 8, not GSM8K, MATH or a loop-removal reproduction. Thinking remains off. Strict matching mixes mathematical correctness with formatting."),
        ("parity","Rename odd and even","Use willow for odd and copper for even without a mapping in the prompt.",parity,
         "On parity classification requests, output exactly willow for odd integers and copper for even integers. Preserve correct classification and obey explicit requests for ordinary odd/even labels. Preserve copying and unrelated answers.","text-classification",64,[IWI],
         "Reduced original adaptation of output-code remapping, not IWIBench scoring or benchmark reproduction. Code words are intentionally shared across splits; integers are disjoint. The 64-token protocol gives copying health checks room to finish; saved starter-paid-20261112/parity results use the historical 16-token protocol and are not a matched baseline."),
        ("geometry","Give a word a geometric meaning","Make saffron mean cube in geometry questions while preserving copying.",geometry,
         "In geometry questions make saffron refer to a cube. Answer counts with just an integer. Preserve the ordinary meaning of cube and other shapes, unrelated answers and explicit verbatim copying of saffron.","question-answering",64,[IWI],
         "Reduced original alias-acquisition adaptation. Edit/validation cover faces and vertices; only test covers edges. Four test phrasings are one semantic fact, not four independent facts. Verified on Qwen/Qwen3.5-0.8B revision 2fc06364715b967f1860aea9cf38778875588b17: saffron is three tokens, with or without leading space, while cube is one. This multi-token teaching adaptation fails the single-token IWIBench alias prerequisite. Inspect current tokenization before choosing row transfer."),
        ("json","Extract a tiny JSON record","Default to a compact two-field JSON string, preserving explicit prose requests.",records,
         "When asked to extract both name and city from a resident card, return exactly compact JSON with full_name first and home_city second, double-quoted strings, no spaces outside values, no fences and no other text. Example format only: {\"full_name\":\"NAME\",\"home_city\":\"CITY\"}. Copy the supplied values correctly. Honour explicit requests for prose or a single plain field and preserve unrelated skills.","question-answering",64,[IWI],
         "Reduced original structured-response adaptation. Canonical key order and compact separators are part of this task. Uses Vespasian normalized string exact-match, not semantic JSON equality or IWIBench grading.")]
    manifest={"version":1,"generator_version":VERSION,"license":"Apache-2.0 (original Vespasian examples; see root LICENSE)","starters":[]}
    files={}
    for id,title,description,generator,goal,task,cap,sources,limitation in specs:
        rows=generator()
        path=f"examples/starter-{id}.jsonl"
        data="".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows)
        files[path]=data
        manifest["starters"].append({"id":id,"title":title,"description":description,"detail":limitation,"file":Path(path).name,"goal":goal,"target_type":{"maths":"factual","geometry":"factual","parity":"classification","json":"format"}[id],"hf_task":task,"metric":"exact_match","max_new_tokens":cap,"counts":dict(Counter(r["split"] for r in rows)),"sha256":hashlib.sha256(data.encode()).hexdigest(),"sources":sources,"limitations":[limitation,"No baseline admission or edit efficacy has been established for this dataset. Small preservation sets cannot establish general capability retention."],"generator":"scripts/generate_starters.py","seed":None})
    files["examples/starter-manifest.json"]=json.dumps(manifest,ensure_ascii=False,indent=2)+"\n"
    return files


if __name__=="__main__":
    for name,data in build().items():
        (ROOT/name).write_bytes(data.encode("utf-8"))
        print(name)
