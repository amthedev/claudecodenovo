function doPost(event) {
  var data = JSON.parse(event.postData.contents || "{}");
  var sourceId = String(data.source_folder || "");
  var destinationId = String(data.destination_folder || "");
  var pattern = String(data.file_pattern || "").toLowerCase();
  var action = String(data.action || "organize");

  if (action === "create_folder") {
    var parent = sourceId ? DriveApp.getFolderById(sourceId) : DriveApp.getRootFolder();
    var folder = parent.createFolder(destinationId || data.name || "Nova pasta");
    return response({ ok: true, folder_id: folder.getId(), folder_name: folder.getName() });
  }

  if (!sourceId || !destinationId) {
    return response({ ok: false, error: "Informe as pastas de origem e destino." });
  }

  var source = DriveApp.getFolderById(sourceId);
  var destination = DriveApp.getFolderById(destinationId);
  var files = source.getFiles();
  var moved = 0;
  while (files.hasNext()) {
    var file = files.next();
    if (!pattern || file.getName().toLowerCase().indexOf(pattern) !== -1) {
      file.moveTo(destination);
      moved++;
    }
  }
  return response({ ok: true, moved: moved, action: action });
}

function response(payload) {
  return ContentService
    .createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}
