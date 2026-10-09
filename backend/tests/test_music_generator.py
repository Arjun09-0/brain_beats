import numpy as np
from backend.app.music.generator import AGITATED_LATENT,CALM_LATENT,FiLMMusicGenerator,generate_midi,midi_metrics,music_controls

def test_latent_controls_are_numerically_distinct(tmp_path):
 import torch
 checkpoint=tmp_path / "demo.pt"; model=FiLMMusicGenerator(); torch.save({"model_state":model.state_dict(),"model_config":{"hidden_size":48,"latent_dim":12}},checkpoint)
 calm,_=generate_midi(CALM_LATENT,checkpoint,beats=16); agitated,_=generate_midi(AGITATED_LATENT,checkpoint,beats=16)
 cm,am=midi_metrics(calm),midi_metrics(agitated)
 assert am["tempo_bpm"] - cm["tempo_bpm"] >= 50
 assert am["note_density"] - cm["note_density"] >= 2

def test_control_contract_rejects_wrong_size():
 try: music_controls(np.zeros(11))
 except ValueError: pass
 else: raise AssertionError("expected dimensionality validation")
