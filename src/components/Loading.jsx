const Loading = ({ color = '#ffffff', size = 56 }) => (
    <div
        className="loading-spinner"
        role="status"
        aria-label="Loading"
        style={{
            '--loading-color': color,
            '--loading-size': `${size}px`
        }}
    />
)

export default Loading