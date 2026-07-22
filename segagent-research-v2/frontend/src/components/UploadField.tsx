import { useState, type ChangeEvent, type DragEvent } from 'react';
import { Check, Loader2, Upload, type LucideIcon } from 'lucide-react';

interface Props {
  id: string;
  title: string;
  hint: string;
  value?: string;
  icon?: LucideIcon;
  multiple?: boolean;
  disabled?: boolean;
  busy?: boolean;
  actionLabel?: string;
  onFiles: (files: File[]) => void | Promise<void>;
}

export default function UploadField({
  id,
  title,
  hint,
  value,
  icon: Icon = Upload,
  multiple = false,
  disabled = false,
  busy = false,
  actionLabel,
  onFiles,
}: Props) {
  const [dragging, setDragging] = useState(false);

  const send = (files: File[]) => {
    if (!disabled && !busy && files.length) void onFiles(files);
  };

  const change = (event: ChangeEvent<HTMLInputElement>) => {
    send(Array.from(event.currentTarget.files || []));
    event.currentTarget.value = '';
  };

  const drop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    send(Array.from(event.dataTransfer.files));
  };

  return (
    <div className="upload-field">
      <input
        id={id}
        className="visually-hidden"
        type="file"
        accept=".nii,.nii.gz"
        multiple={multiple}
        disabled={disabled || busy}
        onChange={change}
      />
      <label
        htmlFor={id}
        className={`upload-card${dragging ? ' is-dragging' : ''}${value ? ' has-file' : ''}`}
        aria-disabled={disabled || busy}
        onDragEnter={event => {
          event.preventDefault();
          if (!disabled && !busy) setDragging(true);
        }}
        onDragOver={event => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        <span className="upload-icon" aria-hidden="true">
          {busy ? <Loader2 className="spin" size={19} /> : value ? <Check size={19} /> : <Icon size={19} />}
        </span>
        <span className="upload-copy">
          <strong>{busy ? `${title}…` : title}</strong>
          <span>{value || hint}</span>
        </span>
        <span className="upload-action" aria-hidden="true">
          {actionLabel || (value ? 'Change' : 'Choose')}
        </span>
      </label>
    </div>
  );
}
