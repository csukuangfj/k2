import k2
s = '''
0 1 2 0.2
0 2 -1 0.1
0 1 1 0.3
2
'''
fsa = k2.Fsa.from_str(s)
fsa.draw('before-connect.svg', title='Before connect')
connected = k2.connect(fsa)
connected.draw('after-connect.svg', title='After connect')
