"""Offline contracts for small, original quick-start teaching datasets."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generate_starters", ROOT / "scripts/generate_starters.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class StarterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((ROOT/"examples/starter-manifest.json").read_text())
        cls.rows={item["id"]:[json.loads(line) for line in (ROOT/"examples"/item["file"]).read_text().splitlines()] for item in cls.manifest["starters"]}

    def test_reproducible_files_and_manifest(self):
        for path,data in generator.build().items():
            self.assertEqual((ROOT/path).read_bytes(),data.encode("utf-8"))
            self.assertNotIn(b"\r", (ROOT/path).read_bytes())
            self.assertTrue((ROOT/path).read_bytes().endswith(b"\n"))
        self.assertEqual(self.manifest["version"],1)
        self.assertEqual(set(self.rows),{"maths","parity","geometry","json"})
        for entry in self.manifest["starters"]:
            path=ROOT/"examples"/entry["file"]
            self.assertEqual(entry["sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(entry["counts"],dict(Counter(r["split"] for r in self.rows[entry["id"]])))
            self.assertEqual(entry["metric"],"exact_match")
            self.assertEqual(entry["max_new_tokens"], {"maths":128,"parity":16,"geometry":64,"json":64}[entry["id"]])

    def test_unique_prompts_and_four_splits(self):
        for rows in self.rows.values():
            self.assertTrue(18<=len(rows)<=24)
            self.assertEqual({r["split"] for r in rows},{"edit","validation","test","preserve"})
            normalized=[" ".join(r["prompt"].lower().split()) for r in rows]
            self.assertEqual(len(set(normalized)),len(rows))
            self.assertTrue(all(r["target"] and r["prompt"] for r in rows))

    def test_math_answers_and_pair_leakage(self):
        pairs=set()
        for rows in self.rows.values():
            local=set()
            for r in rows:
                c=r["case"]
                if "operands" not in c:
                    continue
                op=c["kind"]; numbers=c["operands"]
                if op=="multiply_subtract":
                    a,b,n=numbers
                    expected=a*b-n
                    pair=tuple(sorted((a,b)))
                    self.assertNotIn(pair,pairs)
                    pairs.add(pair)
                elif op=="addition": expected=sum(numbers)
                elif op=="subtraction": expected=numbers[0]-numbers[1]
                elif op=="multiplication": expected=numbers[0]*numbers[1]
                else: self.fail(op)
                self.assertEqual(r["target"],str(expected))
                signature=(op,tuple(sorted(numbers)) if op in {"addition","multiplication"} else tuple(numbers))
                self.assertNotIn(signature,local)
                local.add(signature)

    def test_balanced_parity_and_disjoint_integers(self):
        seen=set()
        rows=self.rows["parity"]
        for split in ("edit","validation","test"):
            selected=[r for r in rows if r["split"]==split]
            counts=Counter(r["target"] for r in selected)
            self.assertEqual(counts["willow"],counts["copper"])
            self.assertTrue(all("willow" not in r["prompt"] and "copper" not in r["prompt"] for r in selected))
        for r in rows:
            if "integer" not in r["case"]: continue
            n=r["case"]["integer"]
            self.assertNotIn(n,seen); seen.add(n)
            labels=("odd","even") if r["split"]=="preserve" else ("willow","copper")
            self.assertEqual(r["target"],labels[0] if n%2 else labels[1])
        self.assertEqual(sum(r["case"]["kind"]=="ordinary_parity" for r in rows),2)

    def test_geometry_heldout_property_and_controls(self):
        for r in self.rows["geometry"]:
            if r["case"]["kind"]!="alias_geometry": continue
            prop=r["case"]["property"]
            self.assertEqual(r["target"],str({"flat faces":6,"vertices":8,"edges":12}[prop]))
            self.assertEqual(prop=="edges",r["split"]=="test")
            self.assertNotIn("cube",r["prompt"])
        self.assertTrue(any(r["target"]=="saffron" and r["split"]=="preserve" for r in self.rows["geometry"]))

    def test_json_targets_canonical_and_records_disjoint(self):
        names=set(); cities=set()
        for r in self.rows["json"]:
            c=r["case"]
            if "name" not in c: continue
            self.assertNotIn(c["name"],names); names.add(c["name"])
            self.assertNotIn(c["city"],cities); cities.add(c["city"])
            if c["kind"]!="record": continue
            parsed=json.loads(r["target"])
            self.assertEqual(list(parsed),["full_name","home_city"])
            self.assertEqual(parsed,{"full_name":c["name"],"home_city":c["city"]})
            self.assertEqual(r["target"],json.dumps(parsed,separators=(",",":")))
            self.assertNotIn("JSON",r["prompt"])
            self.assertNotIn("full_name",r["prompt"])
        for split in ("edit","validation","test"):
            self.assertTrue(any(r["split"]==split and r["case"]["kind"]=="record" for r in self.rows["json"]))


if __name__=="__main__":
    unittest.main()
