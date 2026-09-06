"""
Ponto de integração da SUA IA (deepfake + identidade correta).
"""
from pathlib import Path
from typing import List, Tuple


def verify_frames(frames: List[Path]) -> Tuple[bool, bool, float]:
    """
    TODO: substituir pela chamada real à sua IA.

    Recebe a lista de frames gerados pelo script2 e deve retornar uma tupla:
        (is_deepfake, is_correct_person, confidence)

    Exemplo real:
        from minha_ia import analisar
        resultado = analisar(frames)
        return resultado.is_deepfake, resultado.is_correct_person, resultado.confidence
    """
    # Placeholder: sempre aprova, só para validar o fluxo ponta a ponta.
    is_deepfake = False
    is_correct_person = True
    confidence = 0.99
    return is_deepfake, is_correct_person, confidence
