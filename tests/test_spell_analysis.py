"""Spell matching and context arithmetic checked on small explicit examples."""
import unittest
import numpy as np

from code.spell_analysis import load_patterns,match_spells,primary,spell_context_matrix
from code.vocabulary_map import ROOT


class SpellAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,cls.lookup=load_patterns(ROOT/'config/spells.json')

    def test_longest_boundary_matching_keeps_repeated_mentions_without_double_counting(self):
        tokens,hits=match_spells('LUMOS MAXIMA! Lumos. Lumos maxima! Alohomorax.', 'movies',self.lookup)
        self.assertEqual([h['spell'] for h in hits],['Lumos Maxima','Lumos','Lumos Maxima'])
        self.assertEqual([(h['start'],h['stop']) for h in hits],[(0,2),(2,3),(3,5)])
        self.assertEqual([h['family'] for h in hits],['Lumos']*3)

    def test_explicit_ocr_and_film_variants_are_separate_from_strict_counts(self):
        _,hits=match_spells('Stupefyl! Crucio!','books',self.lookup)
        self.assertEqual(sum(primary(h) for h in hits),2)
        self.assertEqual(sum(primary(h,True) for h in hits),1)
        _,film=match_spells('Stupefyl! Avada Kadavra!','movies',self.lookup)
        self.assertEqual([h['spell'] for h in film],['Avada Kedavra'])
        self.assertFalse(primary(film[0],True))
        self.assertFalse(match_spells('crucial imperiously avada... expecto...', 'books',self.lookup)[1])

    def test_names_and_deliberate_pronunciation_are_not_primary_incantations(self):
        _,hits=match_spells('The Imperious Curse. Imperio! Wingardrium Leviosar!', 'movies',self.lookup)
        self.assertEqual([h['kind'] for h in hits],['name','incantation','pronunciation'])
        self.assertEqual(sum(primary(h) for h in hits),1)
        _,other=match_spells('Cruciatus and Avada Kedavra, the Killing Curse.', 'books',self.lookup)
        self.assertEqual(sum(primary(h) for h in other),1)
        self.assertEqual(sum(h['kind']=='name' for h in other),2)

    def test_context_window_surrounds_whole_phrase_and_masks_other_spells(self):
        tokens=['love','avada','kedavra','power','power','accio','wand']
        def event(spell,start,stop):
            return {'kind':'incantation','spell':spell,'start':start,'stop':stop,'medium':'books',
                    'record_index':1,'tokens':tokens,'masked':{1,2,5}}
        counts,spells,words,supports=spell_context_matrix([event('Avada Kedavra',1,3),event('Accio',5,6)],window=1,min_word_count=1)
        row=spells.index('Avada Kedavra')
        self.assertEqual(counts[row,words.index('love')],1)
        self.assertEqual(counts[row,words.index('power')],1)
        self.assertEqual(counts[row,words.index('wand')],0)
        self.assertEqual(counts[:,words.index('accio')].sum(),0)
        self.assertEqual(counts[spells.index('Accio'),words.index('power')],1)
        self.assertEqual(len(supports['Accio','wand']),1)

    def test_context_eligibility_counts_a_multi_spell_record_once(self):
        tokens=['accio','wand','accio']
        base={'kind':'incantation','spell':'Accio','medium':'books','record_index':1,'tokens':tokens,'masked':{0,2}}
        events=[dict(base,start=0,stop=1),dict(base,start=2,stop=3)]
        counts,spells,words,_=spell_context_matrix(events,window=1,min_word_count=2)
        self.assertNotIn('wand',words)
        events.append(dict(base,record_index=2,start=0,stop=1,tokens=['accio','wand'],masked={0}))
        counts,spells,words,supports=spell_context_matrix(events,window=1,min_word_count=2)
        self.assertEqual(counts[0,words.index('wand')],3)
        self.assertEqual(len(supports['Accio','wand']),2)

    def test_partial_repetition_is_not_a_self_context(self):
        event={'kind':'incantation','spell':'Expecto Patronum','medium':'books','record_index':1,
               'tokens':['expecto','expecto','patronum','wand'],'start':1,'stop':3,'masked':{1,2}}
        matrix,_,words,_=spell_context_matrix([event],min_word_count=1)
        self.assertEqual(matrix[0,words.index('expecto')],0)
        self.assertEqual(matrix[0,words.index('wand')],1)


if __name__=='__main__':unittest.main()
