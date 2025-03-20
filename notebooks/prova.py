import sys
sys.path.append('C:\\SET.IIT\\V\\p\\ddf_p\\ddf_p')

import ddf.common as co
import ddf.hrl as hrl

imp = dict(cartella='tmp', nome='eig', parla=3, campi_post=[])
mdg, dati = hrl.vg14('D', 8, grid_scale=0.25, tipo_griglia='triangolare')
pb = hrl.HRL(mdg, dati, imp)
pb.init()
# pb.steady_state()
# eig = pb.eig(k=5, ks=1)
# print(f'Scenario D8. Primi 5 autovalori: ', eig.vals)
# pb.esporta(stato=1, pkl=1)
# print('Dettagli nella cartella tmp/eig')
