# smart_selection.py
import numpy as np
import math

class SmartTileSelector:
    def __init__(self, class_history=None):
        """
        Initialise le selecteur.
        class_history: dictionnaire des occurrences par classe {class_id: count} (Nc)
        """
        self.class_history = class_history if class_history is not None else {}

    def compute_box_uncertainty(self, conf):
        """
        Eq 1: U(b) = 1 - (P_best - P_second)
        Approximation: YOLOv8 ne retourne que P_best (conf).
        Nous approximons l'ecart avec la 2eme classe la plus probable.
        """
        p_best = float(conf)
        p_second = p_best * 0.8  # Heuristique d'ecart
        u_b = 1.0 - (p_best - p_second)
        return u_b

    def compute_class_rarity(self, cls_id):
        """
        Eq 2: W_c = 1 / ln(e + N_c)
        Amortit l'oubli des classes frequentes tout en valorisant les classes rares.
        """
        n_c = self.class_history.get(cls_id, 0)
        w_c = 1.0 / np.log(np.e + n_c)
        return w_c

    def score_tile(self, yolo_result):
        """
        Eq 3: Score(T) = Somme( U(b) * W_c(b) )
        Agrege les scores des objets pour approximer la densite d'information (HALD).
        """
        tile_score = 0.0
        
        # Si aucun objet n'est detecte, le score est 0 (Filtre Early-Exit)
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return 0.0
            
        for box in yolo_result.boxes:
            try:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
            except Exception:
                continue

            # Calculs
            u_b = self.compute_box_uncertainty(conf)
            w_c = self.compute_class_rarity(cls_id)
            
            # Agregation (Densite)
            tile_score += (u_b * w_c)
            
        return tile_score

    def update_history(self, labels_list):
        """
        Met a jour l'historique Nc apres l'annotation humaine au sol.
        labels_list: liste des classes annotees dans la nouvelle tuile.
        """
        for cls_id in labels_list:
            self.class_history[cls_id] = self.class_history.get(cls_id, 0) + 1