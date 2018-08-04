def find(s, m):
    i = 0
    posi = []
    while i < len(s):
        p = s.find(m, i)
        if p >= 0:
            posi.append(p)
            i = p + len(m)
        else:
            break

    return posi

po = find("北京天安门在北京地铁1号线上，北京地铁很堵", "北京")

print(po)