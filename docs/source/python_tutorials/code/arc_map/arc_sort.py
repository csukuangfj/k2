import k2
s = '''
0 2 -1 0.1
0 1 2 0.2
0 1 1 0.3
2
'''
fsa = k2.Fsa.from_str(s)
fsa.draw('before-arc-sort.svg', title='Before arc-sort')
sorted = k2.arc_sort(fsa)
sorted.draw('after-arc-sort.svg', title='After arc-sort')
