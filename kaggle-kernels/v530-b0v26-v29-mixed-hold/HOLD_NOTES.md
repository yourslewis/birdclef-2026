# v530 HOLD — B0 v26 + NFNet v29 mixed SED bundle

Do not push this kernel until queued public scores for `v522` and/or `v528` justify spending another dataset/kernel/submission slot.

## Future dataset upload

Prepared local bundle:

```text
artifacts/sed_bundles/sed-b0v26-plus-nfnet-v29-oofblend090010-v1.zip
```

Expected private Kaggle dataset slug:

```text
yourslewis/bc26-sed-b0v26-nfnet-v29-oofblend090010-v1
```

Upload command when justified:

```bash
python3 scripts/upload_kaggle_dataset_bearer.py \
  --file artifacts/sed_bundles/sed-b0v26-plus-nfnet-v29-oofblend090010-v1.zip \
  --slug bc26-sed-b0v26-nfnet-v29-oofblend090010-v1 \
  --title "BC26 SED B0v26 NFNet v29 OOF Blend 090010" \
  --description "BirdCLEF 2026 mixed-config TorchScript SED bundle: B0 v26 all-files 0.90 + NFNet v29 0.10" \
  --file-description "TorchScript mixed-config SED bundle zip"
```

Then push this kernel via Bearer/Kaggle workflow and add it after current focus queue, not ahead of already-complete candidates.

## Current validation

- Kernel script py_compile passed.
- Underlying bundle smoke loaded all 6 models and produced `12 x 235`, no NaNs, `5.020s/file` on one real train soundscape.
- This scaffold is based on the public-best `v517` taxon-gated axis, not the older v508/v522 axis.
