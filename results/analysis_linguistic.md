# Linguistic analysis of std->dialect generation

## Region confusion matrix (rows=target, cols=marker-detected region)

| target \ detected | northern | central | southern | none |
|---|---|---|---|---|
| northern | 91 | 79 | 16 | 15 |
| central | 167 | 850 | 39 | 10 |
| southern | 98 | 96 | 126 | 10 |

## Top learned standard→dialect substitutions per region


**northern**: nhỉ→nhờ (28), nhỉ→nhể (18), nhỉ→nhở (12), giàu→giầu (11), trời→giời (10), chưa→chửa (9), gãy→gẫy (7), mẹ→u (5), mẹ→bu (4), nhỉ→nhề (4), màu→mầu (4), nhảy→nhẩy (3)

**central**: này→ni (15), vậy→rứa (10), nào→mô (5), gì→chi (4), chứ→chơ (4), đâu→mô (4), giờ→dừ (3), sao→răng (3), nay→ni (3), đó→nớ (2), nữa→nựa (2), tôi→tui (2)

**southern**: về→dìa (5), không→hông (5), thật→thiệt (3), xem→coi (3), đắt→mắc (3), này→nè (1), chăn→mền (1), buồn→mắc (1), chạnh→nhạnh (1), bạn→ní (1), ngay→liền (1), chứ→chớ (1)


## Copy rate per region

- northern: n=201, copy_rate=9.0%
- central: n=1066, copy_rate=0.5%
- southern: n=330, copy_rate=1.8%