"""Known-family exclusion shared by training, monitoring and final inference."""
import collections
import json
import torch


class FamilyPolicy:
    def __init__(self, path, donor_fonts):
        self.manifest = json.loads(path.read_text())
        self.version = self.manifest['version']
        self.groups = self.manifest['family_by_font']
        members = self.manifest['members']
        assert {f:g for g,fs in members.items() for f in fs} == self.groups
        self.fonts = list(donor_fonts)
        indices = collections.defaultdict(list)
        for i,font in enumerate(self.fonts):
            indices[self.family(font)].append(i)
        self.indices = {g:torch.tensor(ii,dtype=torch.long) for g,ii in indices.items()}
        self.calls = self.removed = self.violations = 0
        self.minimum_eligible = None

    def family(self, font):
        if font not in self.groups:
            raise ValueError('Unknown font family; refusing donor selection: '+font)
        return self.groups[font]

    def exclude(self, font, valid):
        # This mask is applied BEFORE softmax, truncation, or top-k. Query fonts
        # in val/test need not be present in the train donor bank.
        group = self.family(font)
        valid = valid.clone()
        ii = self.indices.get(group)
        if ii is not None:
            self.removed += int(valid[ii].sum())
            valid[ii] = False
        count = int(valid.sum())
        if count == 0:
            raise RuntimeError('No eligible unrelated-family donor: '+font)
        self.calls += 1
        self.minimum_eligible = count if self.minimum_eligible is None else min(self.minimum_eligible,count)
        return valid

    def verify(self, font, selected):
        bad = [d for d in selected if self.family(d)==self.family(font)]
        if bad:
            self.violations += 1
            raise RuntimeError('Same-family donor survived alpha selection: '+repr((font,bad)))

    def snapshot(self):
        return dict(version=self.version,calls=self.calls,masked_family_candidates=self.removed,
                    minimum_eligible=self.minimum_eligible,violations=self.violations)
