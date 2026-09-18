"""Authorized 20k budget; continuous extension for the already-running A arm."""
import math

def original_factor(update,total=10000):
    if update<=500:return update/500
    if update<=total//2:return 1.
    progress=min(1.,(update-total//2)/(total-total//2))
    return .1+.9*(1+math.cos(math.pi*progress))/2

def schedule_spec(auth,arm):
    if auth['successful_updates']==10000:return 'warmup500/flat5000/cosine10pct10000'
    assert auth['successful_updates']==20000
    if arm=='K4-A':return dict(kind='continuous_cosine_extension',**auth['budget_extension']['A_anchor'],total=20000,floor=.1)
    return dict(kind='warmup_flat_cosine',warmup=500,flat_until=10000,total=20000,floor=.1)

def update_factor(update,auth,arm):
    total=auth['successful_updates']
    if total==10000:return original_factor(update)
    assert total==20000
    if arm!='K4-A':return original_factor(update,total=20000)
    anchor=auth['budget_extension']['A_anchor'];start=anchor['step']
    if update<=start:return original_factor(update)
    progress=min(1.,(update-start)/(total-start))
    return .1+(anchor['factor']-.1)*(1+math.cos(math.pi*progress))/2
