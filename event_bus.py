import asyncio

class EventBus:

    def __init__(self):

        self.clients = set()

    async def connect(self, websocket):

        await websocket.accept()

        self.clients.add(websocket)

    def disconnect(self, websocket):

        self.clients.discard(websocket)

    async def broadcast(self, message):

        disconnected = []

        for client in self.clients:

            try:

                await client.send_json(message)

            except Exception:

                disconnected.append(client)

        for client in disconnected:

            self.disconnect(client)


bus = EventBus()