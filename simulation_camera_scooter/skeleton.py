"""
skeleton.py
===========
Skeletonization and branch pruning utilities.
"""

import cv2
import numpy as np

from config import PRUNE_BRANCH_LEN


def skeletonize_guohall(mask_255):
    """Skeletonize a binary mask using Guo-Hall thinning (falls back to morphological)."""
    try:
        from cv2.ximgproc import thinning, THINNING_GUOHALL
        bin_ = ((mask_255 > 0).astype(np.uint8)) * 255
        skel = thinning(bin_, THINNING_GUOHALL)
        # thinning() may return 0/1 or 0/255 depending on OpenCV version
        # normalize to 0/255 safely (avoids 255*255 overflow)
        return ((skel > 0).astype(np.uint8)) * 255
    except ImportError:
        img = (mask_255 > 0).astype(np.uint8)
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel * 255


def extract_skeleton(bev_binary, trim_px=5):
    """Clean BEV mask and extract its skeleton."""
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(bev_binary, cv2.MORPH_CLOSE, kernel)
    clean = cv2.medianBlur(clean, 5)
    _, binary = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)
    skel = skeletonize_guohall(binary)
    if trim_px > 0:
        skel[:trim_px, :] = 0
        skel[-trim_px:, :] = 0
        skel[:, :trim_px] = 0
        skel[:, -trim_px:] = 0
    return skel


def prune_small_branches(skel, min_len=PRUNE_BRANCH_LEN):
    """Iteratively remove short endpoint branches from a skeleton."""
    s = skel.copy()
    for _ in range(min_len):
        nb = cv2.filter2D((s > 0).astype(np.uint8), -1, np.ones((3, 3), np.uint8))
        endpoints = ((s > 0) & (nb == 2))
        s[endpoints] = 0
    return s


def prune_graph_branches(G, min_branch_length=40):
    """Remove short branches from a networkx skeleton graph."""
    G = G.copy()
    endpoints = [n for n in list(G.nodes) if G.degree[n] == 1]
    to_remove = set()
    for ep in endpoints:
        path = [ep]
        current, prev = ep, None
        total_len = 0.0
        while True:
            nbrs = [n for n in G.neighbors(current) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            total_len += G[current][nxt]["weight"]
            path.append(nxt)
            if G.degree[nxt] != 2:
                break
            prev, current = current, nxt
        if total_len < min_branch_length:
            to_remove.update(path)
    G.remove_nodes_from(to_remove)
    return G
