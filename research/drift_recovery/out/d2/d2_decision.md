# D2 decision

Overall: **NO_GLOBAL_G3_VERDICT_UNDERPOWERED_INVALID_OR_UNAVAILABLE**.

G2 is applied before G3. `NO_G3_VERDICT` is not a corrector kill claim.

| Cell | G2 half-width | Invalid gap draws | Primary moved fraction | Harm AUPRC | G3 verdict |
|---|---:|---:|---:|---:|---|
| SciFact / 8 clusters | 0.0190 | 1.53% | 0.079 | 0.225 | NO_G3_VERDICT |
| SciFact / 16 clusters | 0.0148 | 23.76% | 0.105 | 0.151 | NO_G3_VERDICT |
| NFCorpus / 8 clusters | 0.0275 | 76.15% | 0.094 | 0.269 | NO_G3_VERDICT |
| NFCorpus / 16 clusters | 0.0255 | 60.86% | 0.297 | 0.307 | NO_G3_VERDICT |

## Cell evidence

- SciFact / 8: minimum oracle gap 0.0000; decision `NO_G3_VERDICT`.
  - chelation_vs_cbie: ΔNDCG 0.0117 (95% CI -0.0056, 0.0324); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [0.011388124422792911, 0.005313311145783262, 0.008430846887992338, 0.015986599648441957, 0.017285426377748547]; Holm reject=False.
  - chelation_vs_hubness: ΔNDCG 0.0129 (95% CI -0.0046, 0.0340); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [0.0025799825838426216, 0.022310089492598606, 0.010959940717246708, 0.0157563013031774, 0.013093795695379384]; Holm reject=False.
  - chelation_vs_global_ridge: ΔNDCG 0.0012 (95% CI -0.0072, 0.0110); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [-0.0004230128892388496, 0.005162626649268365, 0.008390559590431002, -0.0033525180912081787, -0.003551937580309583]; Holm reject=False.
  G3 was not interpreted: G2 median bootstrap half-width exceeds the preregistered threshold. Rough G2 query budget: 97.
- SciFact / 16: minimum oracle gap -0.0046; decision `NO_G3_VERDICT`.
  - chelation_vs_cbie: ΔNDCG 0.0076 (95% CI -0.0054, 0.0242); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [0.006521808877937185, 0.005537444697989891, 0.014534868127827338, 0.007269468518169009, 0.004338454466795705]; Holm reject=False.
  - chelation_vs_hubness: ΔNDCG 0.0097 (95% CI -0.0054, 0.0280); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [0.01184930265977846, 0.012149609129437855, 0.010280425788926184, 0.009779311737766583, 0.004377867736243601]; Holm reject=False.
  - chelation_vs_global_ridge: ΔNDCG -0.0017 (95% CI -0.0083, 0.0033); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [-0.0025269143130421945, -0.0004501551633847578, -0.0012823180456837946, 0.0027215066934245247, -0.0070242714676714835]; Holm reject=False.
  G3 was not interpreted: non-positive oracle-gap draws exceed the protocol limit.
- NFCorpus / 8: minimum oracle gap -0.0078; decision `NO_G3_VERDICT`.
  - chelation_vs_cbie: ΔNDCG -0.0177 (95% CI -0.0408, 0.0031); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [-0.017960168709933777, -0.012274750136402224, -0.02126282219090092, -0.01652696480399063, -0.020500173903548524]; Holm reject=False.
  - chelation_vs_hubness: ΔNDCG 0.0123 (95% CI -0.0138, 0.0423); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [0.007284527338562663, 0.013728773659774207, 0.014028727613086933, 0.012756887845851828, 0.01366515019921355]; Holm reject=False.
  - chelation_vs_global_ridge: ΔNDCG 0.0209 (95% CI -0.0034, 0.0517); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [0.0026307120162643027, 0.03340334884967999, 0.034222951986618844, 0.01997029855123489, 0.01437641075924867]; Holm reject=False.
  G3 was not interpreted: G2 median bootstrap half-width exceeds the preregistered threshold. Rough G2 query budget: 203.
- NFCorpus / 16: minimum oracle gap -0.0091; decision `NO_G3_VERDICT`.
  - chelation_vs_cbie: ΔNDCG -0.0199 (95% CI -0.0461, 0.0049); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [-0.011765789858789844, -0.018950308683175043, -0.027184983766660564, -0.025790171544848528, -0.015989536447635455]; Holm reject=False.
  - chelation_vs_hubness: ΔNDCG 0.0116 (95% CI -0.0147, 0.0422); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [0.013109046929823287, 0.011628988744589952, 0.010582375711522873, 0.012061167313944243, 0.010386511964084244]; Holm reject=False.
  - chelation_vs_global_ridge: ΔNDCG 0.0224 (95% CI 0.0001, 0.0484); recovery advantage undefined (95% CI undefined, undefined); seed Δ signs [0.02299791687181496, 0.034898073568510046, 0.021298074979676174, 0.005741991108623057, 0.026849291518632024]; Holm reject=False.
  G3 was not interpreted: G2 median bootstrap half-width exceeds the preregistered threshold. Rough G2 query budget: 174.
