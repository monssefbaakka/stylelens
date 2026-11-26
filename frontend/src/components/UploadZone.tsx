import { useEffect, useRef, useState, type DragEvent } from 'react'
import { ACCEPTED_TYPES } from '../api'

interface Props {
  onFile: (file: File) => void
  compact?: boolean
}

/** Dépôt d'image : glisser-déposer, sélection de fichier ou collage (Ctrl+V) n'importe où sur la page. */
export function UploadZone({ onFile, compact = false }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    function onPaste(event: ClipboardEvent) {
      const file = Array.from(event.clipboardData?.files ?? []).find((f) => f.type.startsWith('image/'))
      if (file) {
        event.preventDefault()
        onFile(file)
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [onFile])

  function onDrop(event: DragEvent) {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files[0]
    if (file) onFile(file)
  }

  return (
    <div
      className={`upload${compact ? ' upload--compact' : ''}${dragging ? ' upload--dragging' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        ref={input}
        type="file"
        accept={ACCEPTED_TYPES.join(',')}
        className="visually-hidden"
        aria-label="Choisir une image"
        data-testid="file-input"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          e.target.value = ''
        }}
      />
      {compact ? (
        <button type="button" className="button button--ghost" onClick={() => input.current?.click()}>
          Changer d'image
        </button>
      ) : (
        <>
          <svg className="upload__icon" viewBox="0 0 48 48" aria-hidden="true">
            <rect x="6" y="10" width="36" height="28" rx="4" fill="none" stroke="currentColor" strokeWidth="2.5" />
            <circle cx="17" cy="20" r="3.5" fill="currentColor" />
            <path d="m10 34 9-9 6 6 5-5 8 8" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
          </svg>
          <p className="upload__title">Déposez la photo d'un article</p>
          <p className="upload__hint">vêtement, chaussure ou accessoire · JPEG, PNG ou WebP · 10 Mo max.</p>
          <button type="button" className="button" onClick={() => input.current?.click()}>
            Choisir une image
          </button>
          <p className="upload__hint">ou collez-la avec Ctrl+V</p>
        </>
      )}
    </div>
  )
}
