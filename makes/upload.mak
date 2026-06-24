UPLOAD_URL = https://realsoft.com/cgi-bin/uploadaa.pl/commit_upload
USER_EMAIL = juha@realsoft.fi

upload:
	@echo "Starting explicit PDF upload..."
	@for doc in $(DOCS); do \
		echo "Uploading $$doc.pdf to $(REMOTE_DIR)..."; \
		curl -s -F "filename=@$$doc.pdf" \
		        -F "folder=$(REMOTE_DIR)" \
		        -F "overwrite=1" \
		        -F "from=$(USER_EMAIL)" \
		        -F "submit=Upload" \
		        $(UPLOAD_URL) | grep -E "Error|succesfully" || true; \
	done


uploadpackages:
	@echo "Starting explicit file upload..."
	@for doc in $(PACKAGES); do \
		echo "Uploading $$doc to $(REMOTE_DIR)..."; \
		curl -s -F "filename=@$$doc" \
		        -F "folder=$(REMOTE_DIR)" \
		        -F "overwrite=1" \
		        -F "from=$(USER_EMAIL)" \
		        -F "submit=Upload" \
		        $(UPLOAD_URL) | grep -E "Error|succesfully" || true; \
	done
