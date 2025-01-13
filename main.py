"""
Layout Parser Main API

Author: Krishna Tulsyan (kt.krishna.tulsyan@gmail.com)
"""

import uvicorn

# from server.modules.main.textualAttribute import CustomCNN

if __name__ == '__main__':
	uvicorn.run('server.app:app', host='0.0.0.0', port=8888, reload=True)
