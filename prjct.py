import sys
from collections import deque
sys.setrecursionlimit(1000000)

data = sys.stdin.read().strip().split()
if not data:
    sys.exit(0)

it = iter(data)
n = int(next(it))
s = int(next(it))
t = int(next(it))

graph = [[] for _ in range(n + 1)]
for _ in range(n - 1):
    a = int(next(it))
    b = int(next(it))
    w = int(next(it))
    graph[a].append((b, w))
    graph[b].append((a, w))

visited = [False] * (n + 1)
parent = [-1] * (n + 1)
parent_w = [0] * (n + 1)

queue = deque([s])
visited[s] = True
while queue:
    u = queue.popleft()
    if u == t:
        break
    for v, w in graph[u]:
        if not visited[v]:
            visited[v] = True
            parent[v] = u
            parent_w[v] = w
            queue.append(v)

result = 0
u = t
while u != s:
    result += parent_w[u]
    u = parent[u]

print(result)
