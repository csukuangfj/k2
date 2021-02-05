import k2
s = '''
0 2 1 0.1
1 3 -1 0.3
2 1 2 0.2
3
'''
fsa = k2.Fsa.from_str(s)
fsa.draw('before-top-sort.svg', title='Before top-sort')
sorted = k2.top_sort(fsa)
sorted.draw('after-top-sort.svg', title='After top-sort')
