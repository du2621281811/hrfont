# Frozen E12c fixed-REF8 results

Higher logit means higher learned font compatibility, not probability or universal visual quality. Font-macro averaging; main evaluator refs always8.

## Final matched test16 ×47chars ×4shots

|model|k1|k2|k4|k8|mean|cosine_diagnostic|gt_diagnostic|images|
|---|---|---|---|---|---|---|---|---|
|I0|0.859423|0.922378|0.927560|0.955788|0.916287|0.717230|0.812716|3008|
|I1|0.867072|0.929838|0.898366|0.964596|0.914968|0.720175|0.812716|3008|
|I2|0.878381|0.967820|0.948079|1.014416|0.952174|0.730430|0.812716|3008|
|I3|0.855854|0.938082|0.898660|0.975421|0.917004|0.719780|0.812716|3008|
|I4|0.887919|0.965967|0.952845|1.017815|0.956136|0.731354|0.812716|3008|

## Matched val192: four Latin chars, I0 fixed parent versus I1–I5@4k

|model|step|k1|k4|k8|mean|cosine_diagnostic|images|
|---|---|---|---|---|---|---|---|
|I0|I0 parent|1.081416|1.118025|1.073117|1.090852|0.758542|192|
|I1|4000|1.109059|1.095372|1.123900|1.109443|0.775054|192|
|I2|4000|1.091730|1.073253|1.115644|1.093543|0.768645|192|
|I3|4000|1.152185|1.130046|1.142360|1.141530|0.780541|192|
|I4|4000|1.104329|1.093988|1.126758|1.108358|0.772179|192|
|I5|4000|1.097466|1.099542|1.137862|1.111623|0.777269|192|

## Test script breakdown, all generation shots

|model|latin|kana|bopomofo|
|---|---|---|---|
|I0|0.907271|0.941362|0.968976|
|I1|0.909172|0.933494|0.943224|
|I2|0.943398|0.976339|1.004027|
|I3|0.908832|0.932278|0.982153|
|I4|0.945665|0.971740|1.048877|

## Paired font bootstrap intervals

|comparison|font_paired_logit_delta|ci95_low|ci95_high|fonts|font_wins|note|
|---|---|---|---|---|---|---|
|test_final:I4-I2|0.003962|-0.013069|0.022062|16|7|Paired font bootstrap, descriptive; small sample and multiple comparisons, no corrected significance claim|
|test_final:I2-I0|0.035887|-0.081574|0.139284|16|10|Paired font bootstrap, descriptive; small sample and multiple comparisons, no corrected significance claim|
|test_final:I3-I1|0.002036|-0.018476|0.027461|16|8|Paired font bootstrap, descriptive; small sample and multiple comparisons, no corrected significance claim|
|val4k:I5-I0|0.020771|-0.068760|0.102613|16|10|Paired font bootstrap, descriptive; small sample and multiple comparisons, no corrected significance claim|
|val4k:I5-I3|-0.029907|-0.071335|0.009368|16|6|Paired font bootstrap, descriptive; small sample and multiple comparisons, no corrected significance claim|
|val4k:I5-I4|0.003265|-0.034449|0.043932|16|7|Paired font bootstrap, descriptive; small sample and multiple comparisons, no corrected significance claim|

