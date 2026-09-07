from rdkit import Chem


def canonicalize_smiles(
    smiles,
    canonical: bool = True,
    isomericSmiles: bool = False,
    drop_errorenous_smiles: bool = False,
):
    try:
        return Chem.MolToSmiles(
            Chem.MolFromSmiles(smiles),
            canonical=canonical,
            isomericSmiles=isomericSmiles,
        )
    except Exception as e:
        if drop_errorenous_smiles:
            return None
        else:
            return smiles
