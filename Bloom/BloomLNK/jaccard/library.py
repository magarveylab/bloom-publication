import pandas as pd

from Bloom.BloomLNK import curdir

library_dir = f"{curdir}/jaccard/tables/"

# module to pks reaction
df = pd.read_csv(f"{library_dir}/module_to_pks_reaction.csv")
module_to_pks_reaction = dict(zip(df.module_tag, df.pks_reaction_tag))

# module to substrate
df = pd.read_csv(f"{library_dir}/module_to_substrate.csv")
module_to_substrate = dict(zip(df.module_tag, df.substrate))

# module to substrate family
df = pd.read_csv(f"{library_dir}/module_to_substrate_family.csv")
module_to_substrate_family = dict(zip(df.module_tag, df.substrate_family))

# smarts to units
df = pd.read_csv(f"{library_dir}/smarts_to_unit.csv")
smarts_to_unit = dict(zip(df.smarts_hash_id, df.unit_id))
unit_id_to_unit = dict(zip(df.unit_id, df.unit))
loose_pks_unit_ids = set(
    df[df["unit"].str.contains("-KS|-KR|-ER|-DH|-AMT")].unit_id.tolist()
)
loose_sugar_unit_ids = set(
    df[df["unit"].str.lower().str.contains("ose")].unit_id.tolist()
)

# unit to module tag
df = pd.read_csv(f"{library_dir}/unit_to_module_tag.csv")
unit_to_module_tag = dict(zip(df.unit_id, df.module_tag))
# unit to pks reaction tag
df = pd.read_csv(f"{library_dir}/unit_to_pks_reaction_tag.csv")
unit_to_pks_reaction_tag = dict(zip(df.unit_id, df.pks_reaction_tag))

# unit to substrate
df = pd.read_csv(f"{library_dir}/unit_to_substrate.csv")
unit_to_substrate = dict(zip(df.unit_id, df.substrate))

# unit to substrate family
df = pd.read_csv(f"{library_dir}/unit_to_substrate_family.csv")
unit_to_substrate_family = dict(zip(df.unit_id, df.substrate_family))

# unit to reaction
df = pd.read_csv(f"{library_dir}/unit_to_reaction.csv")
rule_to_reaction = {}
rule_to_unit = {}
unit_to_rule = {}
for unit_id, rule_id, rxn_id in zip(df.unit_id, df.rule_id, df.reaction_id):
    if rule_id not in rule_to_reaction:
        rule_to_reaction[rule_id] = set()
    if unit_id not in unit_to_rule:
        unit_to_rule[unit_id] = set()
    unit_to_rule[unit_id].add(rule_id)
    rule_to_reaction[rule_id].add(rxn_id)
    rule_to_unit[rule_id] = unit_id

# reaction to ec4
df = pd.read_csv(f"{library_dir}/reaction_to_ec4.csv")
reaction_to_ec4 = {}
ec4_to_reaction = {}
for k, v in zip(df.reaction_id, df.ec4):
    if v not in ec4_to_reaction:
        ec4_to_reaction[v] = set()
    ec4_to_reaction[v].add(k)
    if k not in reaction_to_ec4:
        reaction_to_ec4[k] = set()
    reaction_to_ec4[k].add(v)

# reaction to protein family tag
df = pd.read_csv(f"{library_dir}/reaction_to_protein_family_tag.csv")
reaction_to_protein_family_tag = {}
for k, v in zip(df.reaction_id, df.protein_family_tag):
    if k not in reaction_to_protein_family_tag:
        reaction_to_protein_family_tag[k] = set()
    reaction_to_protein_family_tag[k].add(v)

# unit to sugar tag
df = pd.read_csv(f"{library_dir}/unit_to_sugar_tag.csv")
unit_to_sugar_tag = {}
for u, st in zip(df.unit_id, df.sugar_tag):
    if u not in unit_to_sugar_tag:
        unit_to_sugar_tag[u] = set()
    unit_to_sugar_tag[u].add(st)

# sugar tag to ec4
df = pd.read_csv(f"{library_dir}/sugar_tag_to_ec4.csv")
sugar_tag_to_ec4 = {}
ec4_to_sugar_tag = {}
for k, v in zip(df.sugar_tag, df.ec4):
    if k not in sugar_tag_to_ec4:
        sugar_tag_to_ec4[k] = set()
    if v not in ec4_to_sugar_tag:
        ec4_to_sugar_tag[v] = set()
    ec4_to_sugar_tag[v].add(k)
    sugar_tag_to_ec4[k].add(v)

# unit to tailoring tag
df = pd.read_csv(f"{library_dir}/unit_to_tailoring_tag.csv")
unit_to_tailoring_tag = {}
ec3_to_tailoring_unit = {}
for u, tt, et in zip(df.unit_id, df.tailoring_tag, df.ec3):
    unit_to_tailoring_tag[u] = tt
    if et not in ec3_to_tailoring_unit:
        ec3_to_tailoring_unit[et] = set()
    ec3_to_tailoring_unit[et].add(u)

# protein family tag to gene
df = pd.read_csv(f"{library_dir}/gene_to_protein_family.csv")
protein_family_tag_to_gene = {}
gene_to_protein_family_tag = {}
for k, v in zip(df.protein_family_tag, df.gene):
    gene_to_protein_family_tag[v] = k
    if not k in protein_family_tag_to_gene:
        protein_family_tag_to_gene[k] = set()
    protein_family_tag_to_gene[k].add(v)
